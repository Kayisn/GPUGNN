import threading
from pathlib import Path
from queue import Queue

import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import load_gpu_func
from utils.cuda_partition import gpu_partition_graph


def get_gpu_capabilities():
    """Check GPU capabilities including tensor core support"""
    device = cuda.Device(0)

    # Check for Tensor Core support (SM 7.0 or higher)
    compute_capability = device.compute_capability()
    has_tensor_cores = compute_capability[0] >= 7

    return {
        "has_tensor_cores": has_tensor_cores,
        "compute_capability": compute_capability,
        "total_memory": device.total_memory(),
    }


def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)


def extract_submatrices(adjacency_matrix, feature_matrix, cluster_nodes):
    """Extract submatrices including all necessary connections"""
    nodes_idx = sorted(cluster_nodes)

    # Get all nodes that are connected to the cluster nodes
    connected_nodes = set()
    for node in nodes_idx:
        row = adjacency_matrix[:, [node]].tocsr()
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


class CUDAKernelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.kernel = None
        self.streams = {}
        self.context = None
        self.mod = None

    def init_context(self):
        if self.context is None:
            self.context = cuda.Device(0).make_context()
            # Compile kernel once
            try:
                self.kernel = load_gpu_func("sparse_matmul")
                if self.kernel is None:
                    raise RuntimeError("Failed to compile kernel")
            except Exception as e:
                print(f"Error compiling kernel: {e}")
                if self.context:
                    self.context.pop()
                    self.context = None
                raise

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
        self.block = (16, 16, 1)  # Fixed block size for Maxwell
        self.mod = None
        self.kernel = None
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
                if self.kernel_manager.kernel is None:
                    raise RuntimeError("Kernel compilation failed")
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
            self.kernel = load_gpu_func("sparse_matmul")

            # Set block size for Maxwell architecture
            self.block = (16, 16, 1)

    def process_batch(self, batch_data, index):
        """Process a batch using proper CUDA memory management"""
        try:
            self.init_cuda()

            results = []
            for idx, (sub_adj, sub_feat, nodes_idx, all_nodes) in batch_data:
                try:
                    # Convert to CSR and proper types
                    sub_adj = sub_adj.tocsr().astype(np.float32)
                    sub_feat = sub_feat.tocsr().astype(np.float32)

                    # Extract CSR components
                    A_data = sub_adj.data
                    A_indices = sub_adj.indices
                    A_indptr = sub_adj.indptr
                    B_data = sub_feat.data
                    B_indices = sub_feat.indices
                    B_indptr = sub_feat.indptr

                    # Allocate GPU memory in current context
                    gpu_ptrs = []
                    try:
                        A_data_gpu = cuda.mem_alloc(A_data.nbytes)
                        gpu_ptrs.append(A_data_gpu)
                        A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
                        gpu_ptrs.append(A_indices_gpu)
                        A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
                        gpu_ptrs.append(A_indptr_gpu)
                        B_data_gpu = cuda.mem_alloc(B_data.nbytes)
                        gpu_ptrs.append(B_data_gpu)
                        B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
                        gpu_ptrs.append(B_indices_gpu)
                        B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
                        gpu_ptrs.append(B_indptr_gpu)

                        result_size = sub_adj.shape[0] * sub_feat.shape[1]
                        C_gpu = cuda.mem_alloc(result_size * np.float32().itemsize)
                        gpu_ptrs.append(C_gpu)

                        # Copy data to GPU in current context
                        cuda.memcpy_htod(A_data_gpu, A_data)
                        cuda.memcpy_htod(A_indices_gpu, A_indices)
                        cuda.memcpy_htod(A_indptr_gpu, A_indptr)
                        cuda.memcpy_htod(B_data_gpu, B_data)
                        cuda.memcpy_htod(B_indices_gpu, B_indices)
                        cuda.memcpy_htod(B_indptr_gpu, B_indptr)

                        # Calculate grid size
                        grid_x = int(np.ceil(sub_feat.shape[1] / self.block[0]))
                        grid_y = int(np.ceil(sub_adj.shape[0] / self.block[1]))
                        grid = (grid_x, grid_y)

                        with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                            # Launch kernel using current context's function handle
                            self.kernel(
                                A_data_gpu,
                                A_indices_gpu,
                                A_indptr_gpu,
                                B_data_gpu,
                                B_indices_gpu,
                                B_indptr_gpu,
                                C_gpu,
                                np.int32(sub_adj.shape[0]),
                                np.int32(sub_adj.shape[1]),
                                np.int32(sub_feat.shape[1]),
                                block=self.block,
                                grid=grid,
                            )

                        # Get result in current context
                        result = np.empty((sub_adj.shape[0], sub_feat.shape[1]), dtype=np.float32)
                        cuda.memcpy_dtoh(result, C_gpu)

                        # Include timing in results tuple
                        results.append((idx, result[: len(nodes_idx)], nodes_idx))

                    finally:
                        # Clean up GPU memory in current context
                        for ptr in gpu_ptrs:
                            ptr.free()

                except cuda.Error as e:
                    print(f"CUDA error in batch item {idx}: {e}")
                    continue

            return results

        except Exception as e:
            print(f"Error in process_batch: {e}")
            return []
        finally:
            if self.ctx:
                self.ctx.pop()
                self.ctx = None
                self.kernel = None
                self.mod = None

    def process_clusters(self, cluster_data, index):
        """Process clusters with improved error handling"""
        all_results = {}

        # Process in batches without threading
        batches = [
            list(enumerate(cluster_data[i : i + self.batch_size])) for i in range(0, len(cluster_data), self.batch_size)
        ]

        for batch in batches:
            try:
                batch_results = self.process_batch(batch, index)
                if batch_results:  # Only process if we got results
                    for idx, result, nodes_idx in batch_results:
                        all_results[idx] = (result, nodes_idx)  # Store timing
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
    index = graph_info["index"]
    graph = graph_info["graph"]
    num_nodes = graph_info["num_nodes"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    cuda.init()
    try:
        # Initialize kernel manager first
        kernel_manager = CUDAKernelManager.get_instance()
        try:
            kernel_manager.init_context()
        except Exception as e:
            print(f"Failed to initialize global kernel manager: {e}")
            raise

        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0

        # Break these up into smaller blocks for larger graphs using BFS
        adjacency_matrix = adjacency_matrix.tocsr()
        feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

        with nvtx.annotate(f"decomp {index}", domain=Path(__file__).stem):
            # Calculate batch size dynamically based on GPU SMs
            num_sms = get_num_sms()
            threads_per_sm = 1024  # Adjust based on your GPU architecture
            total_threads = num_sms * threads_per_sm
            threads_per_edge = 1  # Define based on kernel requirements
            batch_size = total_threads // threads_per_edge

            # Define the number of clusters - more conservative estimate
            avg_cluster_size = max(int(np.sqrt(num_nodes)), batch_size)  # Increased minimum size
            num_clusters = max(2, num_nodes // avg_cluster_size)  # Added upper limit

            print(f"Decomposing graph into {num_clusters} clusters")
            try:
                # Try GPU-based clustering first
                clusters = gpu_partition_graph(adjacency_matrix, num_clusters)
            except Exception as e:
                print(f"GPU clustering failed: {e}")
                print("Falling back to CPU-based spectral clustering")

            if not clusters:
                raise RuntimeError("Failed to create clusters")

            # Add size-based filtering
            clusters = [c for c in clusters if len(c) >= 2]  # Remove tiny clusters
            if not clusters:
                # If all clusters were filtered out, create one cluster with all nodes
                clusters = [list(range(num_nodes))]

            print(f"Extracting {len(clusters)} clusters")
            # Prepare cluster data
            cluster_data = []
            for cluster_nodes in clusters:
                sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices(
                    adjacency_matrix, feature_matrix, cluster_nodes
                )
                # Check if sub_adj and sub_feat are valid
                if (
                    sub_adj is None
                    or sub_feat is None
                    or sub_adj.shape[0] == 0
                    or sub_adj.shape[1] == 0
                    or sub_feat.shape[0] == 0
                    or sub_feat.shape[1] == 0
                ):
                    print(f"Skipping cluster due to empty submatrices.")
                    continue
                cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))

        # Process clusters using a GPU pipeline
        pipeline = GPUPipeline(batch_size=4)
        pipeline.init_kernel_manager()
        result_dict = pipeline.process_clusters(cluster_data, index)

        # Combine results with proper error handling
        result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
        node_counts = np.zeros(num_nodes, dtype=np.int32)
        success = False
        for idx in range(len(cluster_data)):
            if idx not in result_dict or result_dict[idx] is None:
                continue

            cluster_result, nodes_idx = result_dict[idx]
            if cluster_result is None:
                continue  # Skip failed clusters

            result[nodes_idx] += cluster_result
            node_counts[nodes_idx] += 1
            success = True

        if not success:
            raise RuntimeError("All clusters failed to process")

        # Average results for overlapping nodes
        mask = node_counts != 0
        result[mask] = result[mask] / node_counts[mask, np.newaxis]

        return result
    except Exception as e:
        print(f"Fatal error: {e}")
        print(f"Error processing graph: {e}")
    finally:
        if kernel_manager and kernel_manager.context:
            try:
                kernel_manager.cleanup()
            except:
                pass
