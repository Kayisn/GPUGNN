import multiprocessing as mp
import os
import threading
import time
from multiprocessing import Pool, Queue
from pathlib import Path
from queue import Queue

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import load_gpu_kernel
from utils.cuda_partition import gpu_partition_graph

# Add new constant for timeout duration (in seconds)
KERNEL_TIMEOUT = 3600  # Adjust this value based on your needs


def create_clusters_metis_bfs_gpu(adjacency_matrix, kernel_manager, feature_matrix=None):
    """Create clusters using GPU-accelerated METIS/BFS hybrid partitioning"""
    if not kernel_manager.context:
        kernel_manager.init_context()

    if not kernel_manager.get_kernel("compute_edge_weights") or not kernel_manager.get_kernel("balanced_bfs"):
        raise RuntimeError("Required kernels not initialized")

    # Ensure feature matrix exists
    if feature_matrix is None:
        feature_matrix = sp.eye(adjacency_matrix.shape[0], format="csr")

    clusters = gpu_partition_graph(adjacency_matrix, kernel_manager, feature_matrix)
    if not clusters:
        raise RuntimeError("Partitioning returned no clusters")

    return clusters


def get_gpu_capabilities():
    """Check GPU capabilities including tensor core support"""
    device = cuda.Device(0)
    attributes = device.get_attributes()

    # Check for Tensor Core support (SM 7.0 or higher)
    compute_capability = device.compute_capability()
    has_tensor_cores = compute_capability[0] >= 7

    return {
        "has_tensor_cores": has_tensor_cores,
        "compute_capability": compute_capability,
        "total_memory": device.total_memory(),
    }


matmul_mod = "sparse_bfs"
# gpu_cap = get_gpu_capabilities()
# if gpu_cap["has_tensor_cores"]:
#     print("Using Tensor Cores for sparse matrix multiplication")
#     matmul_mod = "sparse_bfs_tensor"


def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)


def extract_submatrices(adjacency_matrix, feature_matrix, cluster_nodes):
    """Extract submatrices including all necessary connections"""
    nodes_idx = sorted(cluster_nodes)

    # Get all nodes that are connected to the cluster nodes
    connected_nodes = set()
    for node in nodes_idx:
        row = adjacency_matrix[node].tocsr()
        connected_nodes.update(row.indices)

    # Include both cluster nodes and their neighbors
    all_nodes = sorted(set(nodes_idx) | connected_nodes)

    # Extract the relevant submatrices
    sub_adj = adjacency_matrix[nodes_idx, :][:, all_nodes]
    sub_feat = feature_matrix[all_nodes, :]

    if not nodes_idx or not all_nodes:
        return None, None, None, None

    # Ensure sub_adj and sub_feat are in CSR format
    sub_adj = sub_adj.tocsr()
    sub_feat = sub_feat.tocsr()
    return sub_adj, sub_feat, nodes_idx, all_nodes


def init_worker(adj_data, adj_indices, adj_indptr, feat_data, feat_indices, feat_indptr, shape_info):
    """Initialize worker with shared data"""
    global g_adj_matrix, g_feat_matrix
    g_adj_matrix = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=(shape_info[0], shape_info[1]))
    g_feat_matrix = sp.csr_matrix((feat_data, feat_indices, feat_indptr), shape=(shape_info[2], shape_info[3]))


def parallel_extract_submatrix(cluster_nodes):
    """Worker function that uses global shared matrices"""
    try:
        return extract_submatrices(g_adj_matrix, g_feat_matrix, cluster_nodes)
    except Exception as e:
        print(f"Error in worker process: {e}")
        return None


def parallel_process_clusters(adjacency_matrix, feature_matrix, clusters, num_workers=None):
    """Process clusters in parallel using shared memory for large matrices"""
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, len(clusters))

    # Convert matrices to CSR format once
    adj_matrix = adjacency_matrix.tocsr()
    feat_matrix = feature_matrix.tocsr()

    # Prepare shared data
    shape_info = (adj_matrix.shape[0], adj_matrix.shape[1], feat_matrix.shape[0], feat_matrix.shape[1])

    # Initialize the pool with shared data
    pool = Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            adj_matrix.data,
            adj_matrix.indices,
            adj_matrix.indptr,
            feat_matrix.data,
            feat_matrix.indices,
            feat_matrix.indptr,
            shape_info,
        ),
    )

    try:
        # Process clusters in chunks for better efficiency
        chunk_size = max(1, len(clusters) // (num_workers * 4))
        results = []

        # Use imap_unordered for better performance
        for result in pool.imap_unordered(parallel_extract_submatrix, clusters, chunksize=chunk_size):
            if result is not None:
                sub_adj, sub_feat, nodes_idx, all_nodes = result
                if all(x is not None and x.size > 0 for x in [sub_adj.data, sub_feat.data]):
                    results.append((sub_adj, sub_feat, nodes_idx, all_nodes))

        return results

    finally:
        pool.close()
        pool.join()


def vectorized_extract_submatrices(adjacency_matrix, feature_matrix, clusters):
    """Extract all submatrices efficiently using vectorized operations"""
    cluster_data = []

    # Convert to CSR once for efficiency
    adj_csr = adjacency_matrix.tocsr()
    feat_csr = feature_matrix.tocsr()

    for cluster_nodes in clusters:
        nodes_idx = sorted(cluster_nodes)

        # Fast vectorized row slicing
        cluster_rows = adj_csr[nodes_idx]
        connected_nodes = set(cluster_rows.indices)  # Get all column indices at once

        # Combine sets efficiently
        all_nodes = sorted(set(nodes_idx) | connected_nodes)

        # Extract submatrices using efficient slicing
        sub_adj = cluster_rows[:, all_nodes]
        sub_feat = feat_csr[all_nodes]

        if sub_adj.nnz > 0 and sub_feat.nnz > 0:
            cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))

    return cluster_data


class CUDAKernelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.kernels = {}
        self.streams = {}
        self.context = None
        self.mods = {}

    def init_context(self):
        if self.context is None:
            self.context = cuda.Device(0).make_context()
            try:
                print("Compiling CUDA kernels...")
                # Get function handles with exact names from the CUDA code
                (
                    self.kernels["compute_edge_weights"],
                    self.kernels["balanced_bfs"],
                    self.kernels["spectral_clustering"],
                ) = load_gpu_kernel("partition", "compute_edge_weights", "balanced_bfs", "spectral_clustering")

                print("Successfully loaded all partitioning kernels")

                # Load sparse matmul kernel
                self.kernels["sparse_matmul"] = next(load_gpu_kernel(matmul_mod, "matmul"))
                print("Successfully loaded sparse matmul kernel")

            except Exception as e:
                print(f"Error initializing kernels: {str(e)}")
                if self.context:
                    self.context.pop()
                    self.context = None
                raise

    def get_kernel(self, kernel_name):
        """Get a specific compiled kernel"""
        if (kernel_name not in self.kernels) or (self.kernels[kernel_name] is None):
            print(f"Warning: Kernel '{kernel_name}' not found. Available kernels: {list(self.kernels.keys())}")
        return self.kernels.get(kernel_name)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_stream(self, thread_id=None):
        if thread_id is None:
            thread_id = threading.get_ident()
        if thread_id not in self.streams:
            self.streams[thread_id] = cuda.Stream()
        return self.streams[thread_id]

    def cleanup(self):
        if self.context:
            self.context.pop()
            self.context = None


class GPUPipeline:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.kernel_manager = CUDAKernelManager.get_instance()
        self.queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.block_size = (16, 16, 1)  # Default block size
        self.allocated_memory = {}
        self.ctx = None
        self.device = None
        self.use_async = False  # Flag for async operations
        self.kernel_compiled = False
        self.max_threads_per_block = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        self.block_size = None  # Will be set dynamically
        self.block = (32, 32, 1)  # Changed block size
        self.mod = None
        self.kernel = None
        self.sparse_matmul_kernel = None  # Initialize as None
        try:
            # Check if async operations are available
            self.use_async = hasattr(cuda, "mem_alloc_async")
        except:
            self.use_async = False

    def init_kernel_manager(self):
        """Initialize kernel manager in the correct context"""
        if not self.kernel_compiled:
            try:
                if self.kernel_manager is None:
                    self.kernel_manager = CUDAKernelManager.get_instance()
                self.kernel_manager.init_context()

                # Compile kernel directly
                self.sparse_matmul_kernel = next(load_gpu_kernel(matmul_mod, "matmul"))

                if self.sparse_matmul_kernel is None:
                    raise RuntimeError("Failed to compile sparse matmul kernel")

                self.kernel_compiled = True

            except Exception as e:
                print(f"Failed to initialize kernel manager: {e}")
                raise

    def calculate_grid_size(self, rows, cols):
        """Calculate appropriate grid and block sizes based on matrix dimensions and device limits"""
        max_block_size = int(np.sqrt(self.max_threads_per_block))
        block_x = min(32, max_block_size, cols)  # Typically 32 for coalesced memory access
        block_y = min(32, max_block_size, rows)

        self.block_size = (block_x, block_y, 1)

        grid_x = int(np.ceil(cols / block_x))
        grid_y = int(np.ceil(rows / block_y))

        return (grid_x, grid_y, 1)

    def allocate_gpu_memory(self, stream, *arrays):
        """Allocate GPU memory for multiple arrays with fallback to sync operations"""
        gpu_arrays = []
        try:
            for arr in arrays:
                if self.use_async:
                    gpu_arr = cuda.mem_alloc_async(arr.nbytes, stream)
                    cuda.memcpy_htod_async(gpu_arr, arr, stream)
                else:
                    gpu_arr = cuda.mem_alloc(arr.nbytes)
                    cuda.memcpy_htod(gpu_arr, arr)
                gpu_arrays.append(gpu_arr)
                self.allocated_memory[id(gpu_arr)] = gpu_arr
            return gpu_arrays
        except Exception as e:
            print(f"Memory allocation failed: {e}")
            self.free_gpu_memory(*gpu_arrays)
            return []

    def free_gpu_memory(self, *gpu_arrays):
        """Free GPU memory for multiple arrays"""
        for gpu_arr in gpu_arrays:
            if id(gpu_arr) in self.allocated_memory:
                try:
                    gpu_arr.free()
                    del self.allocated_memory[id(gpu_arr)]
                except cuda.Error:
                    pass

    def init_cuda(self):
        """Initialize CUDA context and device with proper kernel loading"""
        if self.ctx is None:
            cuda.init()
            self.device = cuda.Device(0)
            self.ctx = self.device.make_context()

            # Load kernel in this context
            self.kernel = next(load_gpu_kernel(matmul_mod, "matmul"))

            # Set block size for Maxwell architecture
            self.block = (16, 16, 1)

    def process_batch(self, batch_data):
        """Process a batch using proper CUDA memory management"""
        try:
            if not self.kernel_compiled:
                self.init_kernel_manager()

            results = []

            for idx, (sub_adj, sub_feat, nodes_idx, all_nodes) in batch_data:
                try:
                    # Convert to CSR and proper types
                    sub_adj = sub_adj.tocsr().astype(np.float32)
                    sub_feat = sub_feat.tocsr().astype(np.float32)

                    # Calculate dimensions
                    m, k = sub_adj.shape
                    _, n = sub_feat.shape

                    # Calculate grid dimensions based on output size
                    block = (32, 32, 1)  # Fixed block size
                    grid = ((n + block[0] - 1) // block[0], (m + block[1] - 1) // block[1])

                    # Allocate output array
                    result = np.zeros((m, n), dtype=np.float32)

                    # Allocate GPU memory
                    gpu_ptrs = []
                    try:
                        # Allocate with proper alignment
                        A_data_gpu = cuda.mem_alloc(sub_adj.data.nbytes)
                        A_indices_gpu = cuda.mem_alloc(sub_adj.indices.nbytes)
                        A_indptr_gpu = cuda.mem_alloc(sub_adj.indptr.nbytes)
                        B_data_gpu = cuda.mem_alloc(sub_feat.data.nbytes)
                        B_indices_gpu = cuda.mem_alloc(sub_feat.indices.nbytes)
                        B_indptr_gpu = cuda.mem_alloc(sub_feat.indptr.nbytes)
                        C_gpu = cuda.mem_alloc(result.nbytes)

                        gpu_ptrs.extend(
                            [A_data_gpu, A_indices_gpu, A_indptr_gpu, B_data_gpu, B_indices_gpu, B_indptr_gpu, C_gpu]
                        )

                        # Copy data to GPU
                        cuda.memcpy_htod(A_data_gpu, sub_adj.data)
                        cuda.memcpy_htod(A_indices_gpu, sub_adj.indices)
                        cuda.memcpy_htod(A_indptr_gpu, sub_adj.indptr)
                        cuda.memcpy_htod(B_data_gpu, sub_feat.data)
                        cuda.memcpy_htod(B_indices_gpu, sub_feat.indices)
                        cuda.memcpy_htod(B_indptr_gpu, sub_feat.indptr)

                        # Launch kernel
                        self.sparse_matmul_kernel(
                            A_data_gpu,
                            A_indices_gpu,
                            A_indptr_gpu,
                            B_data_gpu,
                            B_indices_gpu,
                            B_indptr_gpu,
                            C_gpu,
                            np.int32(m),
                            np.int32(k),
                            np.int32(n),
                            block=block,
                            grid=grid,
                        )
                        cuda.Context.synchronize()

                        # Copy result back
                        cuda.memcpy_dtoh(result, C_gpu)

                        results.append((idx, result[: len(nodes_idx)], nodes_idx, 0.0))

                    finally:
                        # Clean up GPU memory
                        for ptr in gpu_ptrs:
                            ptr.free()

                except cuda.Error as e:
                    print(f"CUDA error in batch item {idx}: {e}")
                    continue

            return results

        except Exception as e:
            print(f"Error in process_batch: {e}")
            return []

    def process_clusters(self, cluster_data):
        """Process clusters with improved error handling"""
        all_results = {}

        # Process in batches without threading
        batches = [
            list(enumerate(cluster_data[i : i + self.batch_size])) for i in range(0, len(cluster_data), self.batch_size)
        ]

        for batch in batches:
            try:
                batch_results = self.process_batch(batch)
                if batch_results:  # Only process if we got results
                    for idx, result, nodes_idx, timing in batch_results:
                        all_results[idx] = (result, nodes_idx, timing)  # Store timing
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        return all_results

    def cleanup(self):
        """Enhanced cleanup"""
        self.free_gpu_memory(*self.allocated_memory.keys())
        self.allocated_memory.clear()
        if self.ctx:
            try:
                self.ctx.pop()
            except:
                pass
            self.ctx = None


def execute(graph_info, num_warmup=1):
    # Run tests and collect results
    index = graph_info["index"]
    num_nodes = graph_info["num_nodes"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    adjacency_matrix = nx.to_scipy_sparse_array(graph_info["graph"], format="lil", dtype=np.float32)

    # # Perform multiplication (example using BFS and feature matrix)
    # adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    # for node in graph.nodes:
    #     for neighbor in graph.neighbors(node):
    #         adjacency_matrix[node, neighbor] = 1.0

    # # Break these up into smaller blocks for larger graphs using BFS
    # adjacency_matrix = adjacency_matrix.tocsr()

    try:
        # Initialize kernel manager first
        kernel_manager = CUDAKernelManager.get_instance()
        kernel_manager.init_context()

        # # Calculate batch size dynamically based on GPU SMs
        # num_sms = get_num_sms()
        # threads_per_sm = 1024  # Adjust based on your GPU architecture
        # total_threads = num_sms * threads_per_sm
        # threads_per_edge = 1  # Define based on kernel requirements
        # batch_size = total_threads // threads_per_edge

        # Time the decomposition phase
        with nvtx.annotate(f"decomp {index}", domain=Path(__file__).stem):
            # Try GPU-based clustering
            clusters = create_clusters_metis_bfs_gpu(adjacency_matrix, kernel_manager, feature_matrix)

            # Add size-based filtering
            clusters = [c for c in clusters if len(c) >= 2]  # Remove tiny clusters
            if not clusters:
                # If all clusters were filtered out, create one cluster with all nodes
                clusters = [list(range(num_nodes))]

            print(f"Extracting {len(clusters)} clusters")

            # Use vectorized extraction
            cluster_data = vectorized_extract_submatrices(adjacency_matrix, feature_matrix, clusters)

            if not cluster_data:
                print("No valid clusters extracted, skipping graph")
                return

        # Start timing just the multiplication phase
        with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
            pipeline = GPUPipeline(batch_size=4)
            pipeline.init_kernel_manager()
            result_dict = pipeline.process_clusters(cluster_data)

        # Combine results with proper error handling
        merge_start = time.perf_counter()
        result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
        node_counts = np.zeros(num_nodes, dtype=np.int32)
        cluster_times = []

        success = False
        for idx in range(len(cluster_data)):
            if idx not in result_dict or result_dict[idx] is None:
                continue

            cluster_result, nodes_idx, cluster_time = result_dict[idx]
            if cluster_result is None:
                continue  # Skip failed clusters

            result[nodes_idx] += cluster_result
            node_counts[nodes_idx] += 1
            cluster_times.append(cluster_time)
            success = True

        if not success:
            raise RuntimeError("All clusters failed to process")

        # Average results for overlapping nodes
        mask = node_counts != 0
        result[mask] = result[mask] / node_counts[mask, np.newaxis]

        merge_time = time.perf_counter() - merge_start
        print(f"Merge Time elapsed: { time.perf_counter() - merge_time:.4f} seconds")

        return result

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        print(traceback.format_exc())
        return None
    finally:
        if pipeline:
            pipeline.cleanup()
        if kernel_manager and kernel_manager.context:
            kernel_manager.cleanup()
