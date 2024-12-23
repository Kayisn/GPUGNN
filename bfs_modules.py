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
                from cuda_partition_01_08 import PARTITION_KERNELS
                print("Compiling CUDA kernels...")
                
                # Compile partition kernels with proper options
                self.mods['partition'] = SourceModule(
                    PARTITION_KERNELS,
                    options=[],
                    include_dirs=[]
                )
                
                # Define exact kernel names as they appear in CUDA code
                kernel_names = {
                    'bfs_expand_one_level_kernel': 'bfs_expand_one_level_kernel',
                    'find_edge_cuts': 'find_edge_cuts',
                    'spectral_clustering_kernel': 'spectral_clustering_kernel',
                }
                
                # Load kernels
                for api_name, cuda_name in kernel_names.items():
                    try:
                        self.kernels[api_name] = self.mods['partition'].get_function(cuda_name)
                        print(f"Loaded kernel: {api_name}")
                    except cuda.Error as e:
                        print(f"Failed to load kernel {cuda_name}: {e}")
                
                # Load sparse matmul kernel
                self.mods['sparse_matmul'] = SourceModule(kernel_source)
                self.kernels['sparse_matmul'] = self.mods['sparse_matmul'].get_function(kernel_name)
                print("Loaded sparse matmul kernel")
                
            except Exception as e:
                print(f"Error initializing kernels: {str(e)}")
                if self.context:
                    self.context.pop()
                    self.context = None
                raise
    
    def get_kernel(self, kernel_name):
        """Get a specific compiled kernel with better error handling"""
        if not self.kernels:
            raise RuntimeError("Kernels not initialized. Call init_context() first.")
            
        kernel = self.kernels.get(kernel_name)
        if not kernel:
            available = list(self.kernels.keys())
            raise ValueError(f"Kernel '{kernel_name}' not found. Available kernels: {available}")
            
        return kernel

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