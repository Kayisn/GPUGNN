import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty  # Import Empty explicitly

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import numpy as np
from numpy import random as np_random  # Add this import

import concurrent

from verification import verify_result

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

import os

def create_clusters_metis_bfs_gpu(adjacency_matrix, kernel_manager, feature_matrix=None):
    """Create clusters using GPU-accelerated METIS/BFS hybrid partitioning"""
    try:
        if not kernel_manager.context:
            kernel_manager.init_context()
            
        if not kernel_manager.get_kernel('compute_edge_weights') or \
           not kernel_manager.get_kernel('balanced_bfs'):
            raise RuntimeError("Required kernels not initialized")
        
        from cuda_partition_01_05 import gpu_partition_graph
        
        # Ensure feature matrix exists
        if feature_matrix is None:
            feature_matrix = sp.eye(adjacency_matrix.shape[0], format='csr')
            
        clusters = gpu_partition_graph(adjacency_matrix, kernel_manager, feature_matrix)
        if not clusters:
            raise RuntimeError("Partitioning returned no clusters")
            
        return clusters
        
    except Exception as e:
        print(f"GPU clustering failed: {e}")
        # Print more detailed error information
        import traceback
        print(traceback.format_exc())
        raise

# Set CUDA compiler path before importing pycuda
#os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
#os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']

import pycuda.driver as cuda

def get_gpu_capabilities():
    """Check GPU capabilities including tensor core support"""
    device = cuda.Device(0)
    attributes = device.get_attributes()
    
    # Check for Tensor Core support (SM 7.0 or higher)
    compute_capability = device.compute_capability()
    has_tensor_cores = compute_capability[0] >= 7

    return {
        'has_tensor_cores': has_tensor_cores,
        'compute_capability': compute_capability,
        'total_memory': device.total_memory()
    }

def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)



# CUDA kernel for tensor core operations
TENSOR_CORE_KERNEL = """
#include <mma.h>
using namespace nvcuda;

__global__ void sparse_matmul_tensor(
    float *A_data, int *A_indices, int *A_indptr,
    float *B_data, int *B_indices, int *B_indptr,
    float *C, int num_rows, int num_cols, int num_cols_B
) {
    // Define tile sizes for tensor cores
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Shared memory for the tiles
    __shared__ float a_tile[WMMA_M][WMMA_K];
    __shared__ float b_tile[WMMA_K][WMMA_N];
    
    // Initialize accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols_B) {
        float sum = 0;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];
        
        // Load and multiply using tensor cores where possible
        for (int idx = row_start; idx < row_end; idx += WMMA_K) {
            int k_elements = min(WMMA_K, row_end - idx);
            
            // Load tiles into shared memory
            if (threadIdx.x < k_elements) {
                a_tile[threadIdx.y][threadIdx.x] = A_data[idx + threadIdx.x];
            }
            
            __syncthreads();
            
            // Create matrix fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float> b_frag;
            
            // Load fragments
            wmma::load_matrix_sync(a_frag, &a_tile[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &b_tile[0][0], WMMA_N);
            
            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            
            __syncthreads();
        }
        
        // Store result
        C[row * num_cols_B + col] = sum + acc_frag.x[0];
    }
}
"""

# Add new constant for timeout duration (in seconds)
KERNEL_TIMEOUT = 3600  # Adjust this value based on your needs

gpu_caps = get_gpu_capabilities()
kernel_source = """
__global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, 
                            float *B_data, int *B_indices, int *B_indptr, 
                            float *C, int num_rows, int num_cols, int num_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols_B) {
        float sum = 0.0f;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];
        
        // Coalesced memory access pattern for Maxwell
        for (int idx = row_start; idx < row_end; ++idx) {
            int k = A_indices[idx];
            float a_val = A_data[idx];
            
            int col_start = B_indptr[k];
            int col_end = B_indptr[k + 1];
            
            // Binary search for matching column
            int left = col_start;
            int right = col_end - 1;
            
            while (left <= right) {
                int mid = (left + right) >> 1;
                int bcol = B_indices[mid];
                if (bcol == col) {
                    sum += a_val * B_data[mid];
                    break;
                }
                if (bcol < col) left = mid + 1;
                else right = mid - 1;
            }
        }
        C[row * num_cols_B + col] = sum;
    }
}
"""
kernel_name = "sparse_matmul"

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

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool, Queue, Process, Manager
from functools import partial
import numpy as np

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
    shape_info = (adj_matrix.shape[0], adj_matrix.shape[1],
                 feat_matrix.shape[0], feat_matrix.shape[1])
    
    # Initialize the pool with shared data
    pool = Pool(processes=num_workers, 
               initializer=init_worker,
               initargs=(adj_matrix.data, adj_matrix.indices, adj_matrix.indptr,
                        feat_matrix.data, feat_matrix.indices, feat_matrix.indptr,
                        shape_info))
    
    try:
        # Process clusters in chunks for better efficiency
        chunk_size = max(1, len(clusters) // (num_workers * 4))
        results = []
        
        # Use imap_unordered for better performance
        for result in pool.imap_unordered(parallel_extract_submatrix, 
                                        clusters, 
                                        chunksize=chunk_size):
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
                # Load partition kernels
                from cuda_partition_01_05 import PARTITION_KERNELS
                print("Compiling CUDA kernels...")
                
                # Compile without redefining macros that are already in the source
                self.mods['partition'] = SourceModule(
                    PARTITION_KERNELS,
                    options=[],  # Remove macro definitions
                    include_dirs=[]
                )
                
                # Get function handles with exact names from the CUDA code
                try:
                    # These must match the actual function names in the CUDA code
                    self.kernels['compute_edge_weights'] = self.mods['partition'].get_function('compute_edge_weights_kernel')
                    self.kernels['balanced_bfs'] = self.mods['partition'].get_function('balanced_bfs_kernel')
                    self.kernels['spectral_clustering'] = self.mods['partition'].get_function('spectral_clustering_kernel')
                    
                    print("Successfully loaded all kernels")
                    
                    # Set parameter types using PyCUDA's standard pointer type
                    for k in ['compute_edge_weights', 'balanced_bfs', 'spectral_clustering']:
                        if k in self.kernels and self.kernels[k]:
                            self.kernels[k] = self.mods['partition'].get_function(f"{k}_kernel")
                            print(f"Loaded kernel: {k}")
                    
                    # Load sparse matmul kernel
                    self.mods['sparse_matmul'] = SourceModule(kernel_source)
                    self.kernels['sparse_matmul'] = self.mods['sparse_matmul'].get_function(kernel_name)
                    print("Loaded sparse matmul kernel")
                    
                except Exception as e:
                    print(f"Error getting kernel functions: {e}")
                    raise
                
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
            self.use_async = hasattr(cuda, 'mem_alloc_async')
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
                self.mod = SourceModule(kernel_source)
                self.sparse_matmul_kernel = self.mod.get_function(kernel_name)
                
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
            self.mod = SourceModule(kernel_source)
            self.kernel = self.mod.get_function(kernel_name)
            
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
                    grid = (
                        (n + block[0] - 1) // block[0],
                        (m + block[1] - 1) // block[1]
                    )

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
                        
                        gpu_ptrs.extend([A_data_gpu, A_indices_gpu, A_indptr_gpu,
                                       B_data_gpu, B_indices_gpu, B_indptr_gpu, C_gpu])

                        # Copy data to GPU
                        cuda.memcpy_htod(A_data_gpu, sub_adj.data)
                        cuda.memcpy_htod(A_indices_gpu, sub_adj.indices)
                        cuda.memcpy_htod(A_indptr_gpu, sub_adj.indptr)
                        cuda.memcpy_htod(B_data_gpu, sub_feat.data)
                        cuda.memcpy_htod(B_indices_gpu, sub_feat.indices)
                        cuda.memcpy_htod(B_indptr_gpu, sub_feat.indptr)

                        # Launch kernel
                        self.sparse_matmul_kernel(
                            A_data_gpu, A_indices_gpu, A_indptr_gpu,
                            B_data_gpu, B_indices_gpu, B_indptr_gpu,
                            C_gpu, np.int32(m), np.int32(k), np.int32(n),
                            block=block, grid=grid
                        )
                        cuda.Context.synchronize()

                        # Copy result back
                        cuda.memcpy_dtoh(result, C_gpu)
                        
                        results.append((idx, result[:len(nodes_idx)], nodes_idx, 0.0))

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
            list(enumerate(cluster_data[i:i + self.batch_size]))
            for i in range(0, len(cluster_data), self.batch_size)
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

if __name__ == '__main__':
    cuda.init()
    results = []  # Initialize results list
    pipeline = None  # Initialize pipeline to None
    
    try:
        # Initialize kernel manager first
        kernel_manager = CUDAKernelManager.get_instance()
        try:
            kernel_manager.init_context()
        except Exception as e:
            print(f"Failed to initialize global kernel manager: {e}")
            raise

        # Run tests and collect results
        for graph_info in graphs:
            index = graph_info["index"]
            name = graph_info["name"]
            graph_type = graph_info["type"]
            graph = graph_info["graph"]
            feature_matrix = graph_info["feature_matrix"]
            num_nodes = graph_info["num_nodes"]
            sparsity = graph_info["sparsity"]
            print(f"Testing graph {index}")

            # Perform multiplication (example using BFS and feature matrix)
            aggregated_feature_matrix = feature_matrix.copy()

            free_mem, total_mem = cuda.mem_get_info()
            memory_idle = total_mem - free_mem
            stop_event = threading.Event()
            executor = ThreadPoolExecutor(max_workers=1)


            adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for node in graph.nodes:
                for neighbor in graph.neighbors(node):
                    adjacency_matrix[node, neighbor] = 1.0


            # Break these up into smaller blocks for larger graphs using BFS

            adjacency_matrix = adjacency_matrix.tocsr()

            feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

            time.sleep(0.5)  # Wait for memory thread to start

            # Calculate batch size dynamically based on GPU SMs
            num_sms = get_num_sms()
            threads_per_sm = 1024  # Adjust based on your GPU architecture
            total_threads = num_sms * threads_per_sm
            threads_per_edge = 1  # Define based on kernel requirements
            batch_size = total_threads // threads_per_edge

            try:
                # Time the decomposition phase
                decomp_start = time.perf_counter()
                clusters = None  # Initialize clusters to None
                
                try:
                    # Try GPU-based clustering
                    clusters = create_clusters_metis_bfs_gpu(adjacency_matrix, kernel_manager, feature_matrix)
                except Exception as e:
                    print(f"GPU clustering failed: {e} Need to fix partitioning method")
                    continue  # Skip to next graph on clustering failure
                
                if not clusters:
                    print("No clusters created, skipping graph")
                    continue

                previous_time = time.perf_counter()
                print(f"Decomposition Time elapsed: { previous_time - decomp_start:.4f} seconds")
                t1 = previous_time - decomp_start
                
                if not clusters:
                    raise RuntimeError("Failed to create clusters")
                    
                # Add size-based filtering
                clusters = [c for c in clusters if len(c) >= 2]  # Remove tiny clusters
                if not clusters:
                    # If all clusters were filtered out, create one cluster with all nodes
                    clusters = [list(range(num_nodes))]
                    
                print(f"Extracting {len(clusters)} clusters")
                start_extract = time.perf_counter()
                
                # Use vectorized extraction
                cluster_data = vectorized_extract_submatrices(
                    adjacency_matrix,
                    feature_matrix,
                    clusters
                )
                
                t2 = time.perf_counter() - start_extract
                print(f"Vectorized Cluster Extraction Time: {t2:.4f} seconds")
                print(f"Average time per cluster: {t2/len(clusters):.4f} seconds")
                
                if not cluster_data:
                    print("No valid clusters extracted, skipping graph")
                    continue

                previous_time = time.perf_counter()
                
                decomp_time = time.perf_counter() - decomp_start

                # Start timing just the multiplication phase
                start_event = cuda.Event()
                end_event = cuda.Event()
                start_event.record()

                start_time = time.perf_counter()
                
                pipeline = GPUPipeline(batch_size=4)
                pipeline.init_kernel_manager()
                result_dict = pipeline.process_clusters(cluster_data)

                # display time elapsed
                previous_time = time.perf_counter()
                print(f"PipelineTime elapsed: { previous_time - start_time:.4f} seconds")
                t3 = previous_time - start_time
                
                # Combine results with proper error handling
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

                t4 = time.perf_counter() - previous_time
                print(f"Merge Time elapsed: { time.perf_counter() - previous_time:.4f} seconds")
                previous_time = time.perf_counter()
                

                end_event.record()
                end_event.synchronize()
                mult_time = start_event.time_till(end_event)

                end_time = time.perf_counter() - start_time

                start_time = time.perf_counter()

                # Verify the result
                is_correct = verify_result(result, adjacency_matrix, feature_matrix)

                end_time_cpu = time.perf_counter() - start_time

                if not is_correct:
                    print(f"Graph {name} failed verification.")
                else:
                    # Stop memory tracking and get results
                    stop_event.set()
                    
                    results.append({
                        "graph_index": index,
                        "graph_name": name,
                        "graph_type": graph_type,
                        "method": "pycuda_sparse_simplified_06",
                        "decomposition_time": decomp_time,
                        "multiplication_time": end_time,
                        'decomp': t1,
                        'extract': t2,
                        'mult': t3,
                        'merge': t4,
                        "num_clusters": len(clusters),
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_nodes": num_nodes,
                        "sparsity": sparsity,
                        "is_correct": is_correct
                    })


            except cuda.LaunchError as e:
                print(f"CUDA launch failed: {e}")
                stop_event.set()
                continue
            except Exception as e:
                print(f"Error processing graph {name}: {e}")
                stop_event.set()
                continue
            finally:
                if pipeline:
                    pipeline.cleanup()
                    pipeline = None  # Reset pipeline
        
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if pipeline:
            pipeline.cleanup()
        if kernel_manager and kernel_manager.context:
            try:
                kernel_manager.cleanup()
            except:
                pass
        
        # Save results only if we have any
        if results:
            try:
                # Load existing results or create new ones
                if os.path.exists("gnn_results.json"):
                    with open("gnn_results.json", "r") as f:
                        try:
                            all_results = json.load(f)
                        except json.JSONDecodeError:
                            all_results = []
                else:
                    all_results = []

                # Update results
                for result in results:
                    # Check if the result already exists in the list
                    if any(
                        r["graph_index"] == result["graph_index"] and r["method"] == result["method"]
                        for r in all_results
                    ):
                        # If so, replace the existing result
                        all_results = [
                            r
                            for r in all_results
                            if not (
                                r["graph_index"] == result["graph_index"]
                                and r["method"] == result["method"]
                            )
                        ]
                        all_results.append(result)
                    else:
                        all_results.append(result)

                # Save results
                with open("gnn_results.json", "w") as f:
                    json.dump(all_results, f, indent=4)
                print("Results have been saved to 'gnn_results.json'.")
            except Exception as e:
                print(f"Error saving results: {e}")
