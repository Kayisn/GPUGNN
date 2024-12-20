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

VERBOSE = False

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

# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, num_warmup=2, num_test_runs=5):
    """Modified matrix multiplication function with tensor core support"""
    gpu_caps = get_gpu_capabilities()
    
    # Choose appropriate kernel based on GPU capabilities
    if gpu_caps['has_tensor_cores']:
        kernel_source = TENSOR_CORE_KERNEL
        kernel_name = "sparse_matmul_tensor"
    else:
        kernel_source = """
        __global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, float *B_data, int *B_indices, int *B_indptr, float *C, int num_rows, int num_cols, int num_cols_B) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < num_rows && col < num_cols_B) {
                float sum = 0;
                int row_start = A_indptr[row];
                int row_end = A_indptr[row + 1];
                for (int idx = row_start; idx < row_end; ++idx) {
                    int k = A_indices[idx];
                    int col_start = B_indptr[k];
                    int col_end = B_indptr[k + 1];
                    for (int jdx = col_start; jdx < col_end; ++jdx) {
                        if (B_indices[jdx] == col) {
                            sum += A_data[idx] * B_data[jdx];
                            break;
                        }
                    }
                }
                C[row * num_cols_B + col] = sum;
            }
        }
        """
        kernel_name = "sparse_matmul"
    
    # Compile the selected kernel
    mod = SourceModule(kernel_source)
    sparse_matmul = mod.get_function(kernel_name)
    
    # Adjust block size for tensor cores if available
    if gpu_caps['has_tensor_cores']:
        block_size = (16, 16, 1)  # Optimal for tensor cores
    else:
        block_size = (16, 16, 1)  # Standard block size

    # Ensure A and B are in CSR format
    A_csr = A.tocsr().astype(np.float32)
    B_csr = B.tocsr().astype(np.float32)

    # Extract CSR components
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr

    B_data = B_csr.data
    B_indices = B_csr.indices
    B_indptr = B_csr.indptr

    # Ensure A and B have valid shapes
    if A.shape[0] == 0 or A.shape[1] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        raise ValueError("Input matrices A and B must have non-zero dimensions.")

    # Allocate GPU memory for CSR components
    A_data_gpu = cuda.mem_alloc(A_data.nbytes)
    A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
    A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
    B_data_gpu = cuda.mem_alloc(B_data.nbytes)
    B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
    B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
    C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

    # Copy data to GPU (synchronous)
    cuda.memcpy_htod(A_data_gpu, A_data)
    cuda.memcpy_htod(A_indices_gpu, A_indices)
    cuda.memcpy_htod(A_indptr_gpu, A_indptr)
    cuda.memcpy_htod(B_data_gpu, B_data)
    cuda.memcpy_htod(B_indices_gpu, B_indices)
    cuda.memcpy_htod(B_indptr_gpu, B_indptr)

    grid_size = (
        int(np.ceil(B_csr.shape[1] / 16)),
        int(np.ceil(A_csr.shape[0] / 16)),
        1,
    )

    try:
        # Synchronous execution
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            cuda.Context.synchronize()

        # Timing runs
        start_event = cuda.Event()
        end_event = cuda.Event()
        
        times = []
        for _ in range(num_test_runs):
            start_event.record()
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            end_event.record()
            end_event.synchronize()
            times.append(start_event.time_till(end_event))

        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Synchronous copy back
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        # Free GPU memory
        A_data_gpu.free()
        A_indices_gpu.free()
        A_indptr_gpu.free()
        B_data_gpu.free()
        B_indices_gpu.free()
        B_indptr_gpu.free()
        C_gpu.free()

        return C_dense, mean_time, std_time

    except:
        # Cleanup on error
        try:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()
        except:
            pass
        raise

def create_clusters_spectral(adjacency_matrix, num_clusters):
    """
    Create clusters using Spectral Clustering with handling for disconnected components.
    """
    # Ensure the adjacency matrix is in a proper format
    adjacency_matrix = adjacency_matrix.astype(np.float32)
    
    # Add small self-loops to ensure connectivity
    adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0], format='csr') * 1e-6
    
    # Compute the normalized Laplacian
    n_nodes = adjacency_matrix.shape[0]
    if isinstance(adjacency_matrix, sp.spmatrix):
        # Get degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        # Compute D^(-1/2)
        degrees_inv_sqrt = np.power(degrees, -0.5)
        degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(degrees_inv_sqrt)
        # Normalized Laplacian
        L_norm = sp.eye(n_nodes) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        # Convert to dense for spectral clustering
        L_norm = L_norm.toarray()
    else:
        L_norm = adjacency_matrix

    # Perform spectral clustering with more robust parameters
    clustering = SpectralClustering(
        n_clusters=min(num_clusters, n_nodes - 1),  # Ensure we don't exceed n_nodes-1
        affinity='precomputed_nearest_neighbors',
        assign_labels='kmeans',
        random_state=0,
        n_neighbors=min(10, n_nodes - 1),  # Adjust neighborhood size
        n_init=10  # Multiple initialization for k-means
    )
    
    # Use the normalized Laplacian as the affinity matrix
    labels = clustering.fit_predict(L_norm)

    # Group nodes by cluster labels
    clusters = []
    for i in range(clustering.n_clusters_):
        cluster_nodes = np.where(labels == i)[0].tolist()
        if cluster_nodes:  # Only add non-empty clusters
            clusters.append(cluster_nodes)
    
    return clusters

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

def extract_submatrices_pagerank(adjacency_matrix, feature_matrix, cluster_nodes):
    """
    Extract submatrices for PageRank-based clustering, ensuring proper connectivity
    and handling edge cases for sparse graphs.
    """
    nodes_idx = sorted(cluster_nodes)
    
    # Get two-hop neighborhood to capture more context
    connected_nodes = set()
    second_hop = set()
    
    # First hop
    for node in nodes_idx:
        row = adjacency_matrix[node].tocsr()
        new_neighbors = set(row.indices)
        connected_nodes.update(new_neighbors)
        
        # Second hop - look at neighbors' neighbors
        for neighbor in new_neighbors:
            neighbor_row = adjacency_matrix[neighbor].tocsr()
            second_hop.update(neighbor_row.indices)
    
    # Include both direct connections and second-hop nodes
    # but prioritize immediate neighbors
    all_nodes = sorted(set(nodes_idx) | connected_nodes)
    extended_nodes = sorted(second_hop - set(all_nodes))
    
    # If cluster is too small, include second-hop nodes
    min_size = 16  # Minimum size for efficient GPU processing
    if len(all_nodes) < min_size:
        all_nodes.extend(extended_nodes[:min_size - len(all_nodes)])
    
    # Extract the submatrices
    sub_adj = adjacency_matrix[nodes_idx, :][:, all_nodes]
    sub_feat = feature_matrix[all_nodes, :]
    
    # Add self-loops to ensure numerical stability
    sub_adj = sub_adj + sp.eye(sub_adj.shape[0], sub_adj.shape[1], 
                              format='csr', dtype=np.float32) * 1e-6
    
    # Normalize adjacency matrix
    degrees = np.array(sub_adj.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        deg_inv = np.power(degrees, -0.5)
    deg_inv[np.isinf(deg_inv)] = 0
    deg_mat_inv = sp.diags(deg_inv)
    
    # Symmetric normalization
    sub_adj = deg_mat_inv @ sub_adj @ deg_mat_inv
    
    if not nodes_idx or not all_nodes:
        return None, None, None, None
    
    # Ensure matrices are in CSR format and properly typed
    sub_adj = sub_adj.tocsr().astype(np.float32)
    sub_feat = sub_feat.tocsr().astype(np.float32)
    
    # Print diagnostics
    print(f"Cluster size: {len(nodes_idx)}")
    print(f"Extended size: {len(all_nodes)}")
    print(f"Adjacency matrix shape: {sub_adj.shape}, nnz: {sub_adj.nnz}")
    print(f"Feature matrix shape: {sub_feat.shape}, nnz: {sub_feat.nnz}")
    
    return sub_adj, sub_feat, nodes_idx, all_nodes

class CUDAContextManager:
    def __init__(self):
        self.context = None
        
    def __enter__(self):
        self.context = cuda.Device(0).make_context()
        return self
        
    def __exit__(self, *args):
        if self.context:
            self.context.pop()

def process_cluster(cluster_data):
    """Process a single cluster using sparse matrix multiplication"""
    with CUDAContextManager():
        sub_adj, sub_feat, nodes_idx, all_nodes = cluster_data
        result, cluster_time, _ = sparse_matrix_multiply_pycuda(sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
        # Only return results for the original cluster nodes
        return result[:len(nodes_idx)], cluster_time

def process_cluster_pipelined(cluster_queue, result_dict, lock):
    """Process clusters sequentially with proper context management"""
    with CUDAContextManager():
        while True:
            try:
                idx, (sub_adj, sub_feat, nodes_idx, all_nodes) = cluster_queue.get_nowait()
                try:
                    result, cluster_time, _ = sparse_matrix_multiply_pycuda(
                        sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                    
                    with lock:
                        result_dict[idx] = (result[:len(nodes_idx)], nodes_idx, cluster_time)
                except Exception as e:
                    print(f"Error processing cluster {idx}: {e}")
                    with lock:
                        result_dict[idx] = None
            except Empty:
                break

import threading
from contextlib import contextmanager
import threading
import multiprocessing

class CUDAThreadManager:
    _local = threading.local()
    
    @classmethod
    @contextmanager
    def get_context(cls):
        if not hasattr(cls._local, 'context_count'):
            cls._local.context_count = 0
            cls._local.context = cuda.Device(0).make_context()
        else:
            cls._local.context.push()

        try:
            yield
        finally:
            cls._local.context.pop()

def calculate_matrix_memory(matrix):
    """Calculate actual memory requirements for a sparse matrix"""
    # Account for data array
    data_mem = matrix.data.nbytes
    # Account for indices
    indices_mem = matrix.indices.nbytes
    # Account for indptr
    indptr_mem = matrix.indptr.nbytes
    # Account for shape and other metadata
    metadata_mem = 128  # Conservative estimate
    
    total = data_mem + indices_mem + indptr_mem + metadata_mem
    return total

def process_cluster_worker(args):
    """Process a single cluster with improved memory handling"""
    idx, data = args
    try:
        context = cuda.Device(0).make_context()
        try:
            if VERBOSE:
                sub_adj, sub_feat, nodes_idx, all_nodes = data
                
                # Enhanced memory calculation
                adj_mem = calculate_matrix_memory(sub_adj)
                feat_mem = calculate_matrix_memory(sub_feat)
                
                # Result matrix size (dense)
                result_mem = sub_adj.shape[0] * sub_feat.shape[1] * np.float32().itemsize
                
                # Working memory for GPU operations
                # Account for temporary buffers and alignment
                working_mem = (adj_mem + feat_mem) * 2  # Double for workspace
                alignment_padding = 256  # GPU memory alignment requirement
                
                # Total with alignment considerations
                total_mem = (
                    adj_mem + feat_mem + result_mem + working_mem + 
                    alignment_padding * 4  # Add padding for each allocation
                )
                
                print(f"\nProcessing cluster {idx}:")
                print(f"Cluster size: {len(nodes_idx)} nodes")
                print(f"Extended size: {len(all_nodes)} nodes")
                print(f"Sub-adjacency matrix: {sub_adj.shape}, nnz: {sub_adj.nnz}")
                print(f"Sub-feature matrix: {sub_feat.shape}, nnz: {sub_feat.nnz}")
                
                print(f"Memory requirements:")
                print(f"- Adjacency matrix: {adj_mem/1e6:.2f}MB (nnz: {sub_adj.nnz})")
                print(f"- Feature matrix: {feat_mem/1e6:.2f}MB (nnz: {sub_feat.nnz})")
                print(f"- Result matrix: {result_mem/1e6:.2f}MB")
                print(f"- Working memory: {working_mem/1e6:.2f}MB")
                print(f"- Alignment padding: {(alignment_padding*4)/1e6:.2f}MB")
                print(f"- Total required: {total_mem/1e6:.2f}MB")
                
                # Ensure minimum allocation size
                min_allocation = 1024  # Minimum allocation size in bytes
                if adj_mem < min_allocation or feat_mem < min_allocation:
                    print(f"Warning: Matrices too small, adjusting allocation size")
                    adj_mem = max(adj_mem, min_allocation)
                    feat_mem = max(feat_mem, min_allocation)
                    total_mem = adj_mem + feat_mem + result_mem + working_mem
                
                # Check available memory
                free_mem, total_gpu_mem = cuda.mem_get_info()
                if total_mem > free_mem * 0.8:  # Need at least 20% free memory
                    print(f"Warning: Insufficient memory for cluster {idx}")
                    print(f"Free memory: {free_mem/1e6:.2f}MB, Required: {total_mem/1e6:.2f}MB")
                    return idx, None, None, 0
                    
                print(f"Memory check passed. Processing cluster {idx}")
                

                sub_adj, sub_feat, nodes_idx, all_nodes = data

                # Print detailed matrix information
                print(f"\nProcessing cluster {idx}:")
                print(f"Cluster size: {len(nodes_idx)} nodes")
                print(f"Extended size: {len(all_nodes)} nodes")
                print(f"Sub-adjacency matrix shape: {sub_adj.shape}")
                print(f"Sub-feature matrix shape: {sub_feat.shape}")
                
                # Calculate actual memory requirements
                adj_mem = (sub_adj.data.nbytes + 
                        sub_adj.indices.nbytes + 
                        sub_adj.indptr.nbytes)
                feat_mem = (sub_feat.data.nbytes + 
                        sub_feat.indices.nbytes + 
                        sub_feat.indptr.nbytes)
                result_mem = sub_adj.shape[0] * sub_feat.shape[1] * np.float32().itemsize
                working_mem = max(adj_mem, feat_mem) * 3  # Buffer for GPU operations
                total_mem = adj_mem + feat_mem + result_mem + working_mem
                
                print(f"Memory requirements:")
                print(f"- Adjacency matrix: {adj_mem/1e6:.2f}MB")
                print(f"- Feature matrix: {feat_mem/1e6:.2f}MB")
                print(f"- Result matrix: {result_mem/1e6:.2f}MB")
                print(f"- Working memory: {working_mem/1e6:.2f}MB")
                print(f"- Total required: {total_mem/1e6:.2f}MB")
                
                # Check available memory
                free_mem, total_mem = cuda.mem_get_info()
                if total_mem < free_mem * 0.2:  # Need at least 20% free memory
                    print(f"Warning: Insufficient memory for cluster {idx}")
                    print(f"Free memory: {free_mem/1e6:.2f}MB, Required: {total_mem/1e6:.2f}MB")
                    return idx, None, None, 0
            
                print(f"Memory check passed. Processing cluster {idx}")
            try:
                result, timing, _ = sparse_matrix_multiply_pycuda(
                    sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                if VERBOSE:
                    print(f"Successfully processed cluster {idx}")
                return idx, result[:len(nodes_idx)], nodes_idx, timing
            except cuda.Error as e:
                print(f"CUDA error in cluster {idx}: {str(e)}")
                return idx, None, None, 0
                
        finally:
            context.pop()
    except Exception as e:
        print(f"Error processing cluster {idx}: {e}")
        return idx, None, None, 0

class Pipeline:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.lock = threading.Lock()
        
    def process_batch(self, batch_idx, batch_data):
        with CUDAThreadManager.get_context():
            results = []
            for idx, data in batch_data:
                try:
                    sub_adj, sub_feat, nodes_idx, all_nodes = data
                    result, timing, _ = sparse_matrix_multiply_pycuda(
                        sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                    results.append((idx, result[:len(nodes_idx)], nodes_idx, timing))
                except Exception as e:
                    print(f"Error in batch {batch_idx}, cluster {idx}: {e}")
                    continue
            
            with self.lock:
                self.output_queue.put((batch_idx, results))

    def process_clusters(self, cluster_data, num_workers=2):
        """Process clusters with consistent error handling"""
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_cluster_worker, enumerate(cluster_data))
            
        result_dict = {}
        for idx, cluster_result, nodes_idx, timing in results:
            if cluster_result is not None and nodes_idx is not None:
                result_dict[idx] = (cluster_result, nodes_idx, timing)
            else:
                if VERBOSE:
                    print(f"Cluster {idx} was skipped or failed; excluding from results.")
                result_dict[idx] = None
                
        return result_dict
    
# Add CUDA initialization check
def init_cuda():
    """Initialize CUDA device"""
    try:
        import pycuda.autoinit
        return cuda.Device(0).compute_capability()
    except:
        return None
    
# Add new CUDA kernel for PageRank
PAGERANK_KERNEL = """
__global__ void pagerank_iteration(
    const float *in_rank,
    float *out_rank,
    const int *row_ptr,
    const int *col_idx,
    const float *values,
    const float damping,
    const int num_nodes
) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (node < num_nodes) {
        float sum = 0.0f;
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        
        // Sum contributions from incoming edges
        for (int edge = start; edge < end; edge++) {
            int src = col_idx[edge];
            int src_degree = row_ptr[src + 1] - row_ptr[src];
            if (src_degree > 0) {
                sum += in_rank[src] / src_degree;
            }
        }
        
        // Apply damping factor
        out_rank[node] = (1.0f - damping) / num_nodes + damping * sum;
    }
}
"""

def calculate_pagerank_gpu(adjacency_matrix, damping=0.85, max_iterations=100, tolerance=1e-6):
    """Calculate PageRank scores using GPU acceleration with memory management"""
    
    # Check CUDA availability
    if init_cuda() is None:
        return calculate_pagerank_cpu(adjacency_matrix, damping, max_iterations, tolerance)

    num_nodes = adjacency_matrix.shape[0]
    
    try:
        # Convert to CSR format
        if not sp.isspmatrix_csr(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.tocsr()
            
        # Calculate memory sizes correctly
        rank_size = num_nodes * np.dtype(np.float32).itemsize
        row_ptr_size = (num_nodes + 1) * np.dtype(np.int32).itemsize
        col_idx_size = len(adjacency_matrix.indices) * np.dtype(np.int32).itemsize
        
        # Allocate GPU memory
        rank_current = np.ones(num_nodes, dtype=np.float32) / num_nodes
        rank_next = np.zeros_like(rank_current)
        
        rank_current_gpu = cuda.mem_alloc(rank_size)
        rank_next_gpu = cuda.mem_alloc(rank_size)
        row_ptr_gpu = cuda.mem_alloc(row_ptr_size)
        col_idx_gpu = cuda.mem_alloc(col_idx_size)
        
        # Transfer initial data
        cuda.memcpy_htod(rank_current_gpu, rank_current)
        cuda.memcpy_htod(row_ptr_gpu, adjacency_matrix.indptr.astype(np.int32))
        cuda.memcpy_htod(col_idx_gpu, adjacency_matrix.indices.astype(np.int32))
        
        # Compile and configure kernel
        mod = SourceModule(PAGERANK_KERNEL)
        pagerank_kernel = mod.get_function("pagerank_iteration")
        
        block_size = min(512, num_nodes)
        grid_size = (num_nodes + block_size - 1) // block_size
        
        # Main iteration loop
        for iteration in range(max_iterations):
            pagerank_kernel(
                rank_current_gpu, rank_next_gpu,
                row_ptr_gpu, col_idx_gpu,
                np.float32(damping), np.int32(num_nodes),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            cuda.Context.synchronize()
            
            # Check convergence
            cuda.memcpy_dtoh(rank_next, rank_next_gpu)
            diff = np.abs(rank_next - rank_current).max()
            
            if diff < tolerance:
                break
                
            # Swap buffers correctly
            rank_current_gpu, rank_next_gpu = rank_next_gpu, rank_current_gpu
            rank_current = rank_next.copy()
        
        # Get final results and normalize
        cuda.memcpy_dtoh(rank_current, rank_current_gpu)
        rank_current /= rank_current.sum()
        
        # Cleanup
        for gpu_array in [rank_current_gpu, rank_next_gpu, row_ptr_gpu, col_idx_gpu]:
            gpu_array.free()
            
        return rank_current
        
    except cuda.Error as e:
        print(f"CUDA Error: {e}")
        return calculate_pagerank_cpu(adjacency_matrix, damping, max_iterations, tolerance)



def calculate_pagerank_cpu(adjacency_matrix, damping=0.85, max_iterations=100, tolerance=1e-6):
    """Fallback CPU implementation of PageRank"""
    print("Using CPU PageRank implementation")
    num_nodes = adjacency_matrix.shape[0]
    
    # Initialize scores
    scores = np.ones(num_nodes) / num_nodes
    
    # Convert to CSR for efficient operations
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()
    
    # Calculate degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    
    # Normalize adjacency matrix by degree
    adj_normalized = adjacency_matrix.copy()
    adj_normalized.data /= degrees[adj_normalized.indices]
    
    # PageRank iteration
    for _ in range(max_iterations):
        scores_next = (1 - damping) / num_nodes + damping * adj_normalized.T.dot(scores)
        
        # Check convergence
        if np.abs(scores_next - scores).max() < tolerance:
            break
            
        scores = scores_next.copy()
    
    return scores

def bfs_cluster(adjacency_matrix, start_node, max_edges):
    """Create a cluster using BFS until reaching max_edges"""
    nodes_in_cluster = set()
    edges_in_cluster = 0
    queue = [start_node]
    nodes_in_cluster.add(start_node)
    
    while queue and edges_in_cluster < max_edges:
        current = queue.pop(0)
        row = adjacency_matrix[current].tocsr()
        neighbors = row.indices
        
        for neighbor in neighbors:
            if neighbor not in nodes_in_cluster:
                nodes_in_cluster.add(neighbor)
                queue.append(neighbor)
            edges_in_cluster += 1  # Count all edges, even to existing nodes
            
    return list(nodes_in_cluster)

def create_clusters_pagerank(adjacency_matrix, max_edges_per_cluster, seed=None):
    num_nodes = adjacency_matrix.shape[0]
    try:
        # Calculate PageRank scores with error handling
        pagerank_scores = calculate_pagerank_gpu(adjacency_matrix)
        if pagerank_scores is None:
            print("PageRank calculation failed")
    except Exception as e:
        print(f"Error during PageRank calculation: {e}")
        pagerank_scores = None

    """Create clusters using PageRank-based node selection"""
    rng = np_random.RandomState(seed)  # Create a random number generator
    
    num_nodes = adjacency_matrix.shape[0]
    remaining_edges = set((i, j) for i, j in zip(*adjacency_matrix.nonzero()))
    unvisited = set(range(num_nodes))
    clusters = []
    
    if VERBOSE:
        print("Calculating PageRank scores on GPU...")

    # Calculate PageRank scores
    pagerank_scores = calculate_pagerank_gpu(adjacency_matrix)
    
    while unvisited:
        # Normalize scores for unvisited nodes
        valid_scores = np.zeros(num_nodes)
        for node in unvisited:
            valid_scores[node] = pagerank_scores[node]
        
        if valid_scores.sum() == 0:
            start_node = min(unvisited)  # Fallback to deterministic selection
        else:
            # Normalize to probabilities
            valid_scores = valid_scores / valid_scores.sum()
            start_node = rng.choice(num_nodes, p=valid_scores)  # Use rng instead of np_random
        
        # Rest of the function remains the same
        cluster_nodes = bfs_cluster(adjacency_matrix, start_node, max_edges_per_cluster)
        clusters.append(cluster_nodes)
        
        for node in cluster_nodes:
            row = adjacency_matrix[node].tocsr()
            for neighbor in row.indices:
                remaining_edges.discard((node, neighbor))
        
        unvisited -= set(cluster_nodes)
    
    # Handle remaining edges
    while remaining_edges:
        edge = remaining_edges.pop()
        added = False
        for cluster in clusters:
            if edge[0] in cluster or edge[1] in cluster:
                cluster.extend([edge[0], edge[1]])
                cluster = list(set(cluster))
                added = True
                break
        if not added:
            clusters.append([edge[0], edge[1]])
    
    return clusters

import networkit as nk  # Add this import at the top
import networkx as nx  # Add this import

import networkit as nk  # Add this import at the top
import networkx as nx  # Add this import

def create_clusters_metis(adjacency_matrix, num_clusters):
    """
    Create clusters using METIS-style graph partitioning via NetworKit.
    Runtime complexity: O(|E|) - near linear in the number of edges
    """
    # Convert scipy sparse matrix to NetworkX graph using the correct function
    if hasattr(nx, 'from_scipy_sparse_matrix'):  # older versions
        nx_graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    else:  # newer versions
        nx_graph = nx.from_scipy_sparse_array(adjacency_matrix)
    
    # Convert NetworkX graph to NetworKit graph
    g = nk.nxadapter.nx2nk(nx_graph)
    
    # Run partitioning with PLM (Parallel Louvain Method)
    communities = nk.community.PLM(g).run().getPartition()
    
    # Convert partition to clusters
    num_communities = communities.numberOfSubsets()
    clusters = [[] for _ in range(num_communities)]
    for node in range(g.numberOfNodes()):
        comm_id = communities[node]
        clusters[comm_id].append(node)
    
    # Remove empty clusters and ensure balanced sizes
    clusters = [c for c in clusters if c]
    
    # Ensure we don't exceed desired number of clusters
    if len(clusters) > num_clusters:
        # Merge smallest clusters
        clusters.sort(key=len)
        while len(clusters) > num_clusters:
            smallest = clusters.pop(0)
            clusters[0].extend(smallest)
    
    return clusters

def create_clusters_metis_like(adjacency_matrix, num_clusters):
    """
    Create clusters using METIS-style graph partitioning via NetworKit.
    Runtime complexity: O(|E|) - near linear in the number of edges
    """
    # Convert to NetworkX then NetworKit format
    nx_graph = nx.from_scipy_sparse_array(adjacency_matrix)
    g = nk.nxadapter.nx2nk(nx_graph)
    
    # Run partitioning
    communities = nk.community.PLM(g).run().getPartition()
    
    # Convert results to cluster format
    clusters = [[] for _ in range(communities.numberOfSubsets())]
    for node in range(g.numberOfNodes()):
        comm_id = communities[node]
        clusters[comm_id].append(node)
        
    # Balance cluster sizes
    clusters = [c for c in clusters if c]
    if len(clusters) > num_clusters:
        clusters.sort(key=len)
        while len(clusters) > num_clusters:
            smallest = clusters.pop(0)
            clusters[0].extend(smallest)
    
    return clusters

def create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters):
    """
    Create clusters using GPU-accelerated METIS/BFS hybrid partitioning
    """
    from cuda_partition import gpu_partition_graph
    return gpu_partition_graph(adjacency_matrix, num_clusters)

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Define clustering methods and their parameters
CLUSTERING_METHODS = {
    'metis_bfs_gpu': {
        'func': create_clusters_metis_bfs_gpu,
        'params': {'num_clusters': None},  # Will be set based on graph size
        'name': 'GPU METIS-BFS'
    },
    'metis': {
        'func': create_clusters_metis,
        'params': {'num_clusters': None},  # Will be set based on graph size
        'name': 'METIS'
    },
    'metis-like': {
        'func': create_clusters_metis_like,
        'params': {'num_clusters': None},  # Will be set based on graph size
        'name': 'METIS-Like'
    },
    'pagerank': {
        'func': create_clusters_pagerank,
        'params': {'max_edges_per_cluster': None},
        'name': 'PageRank-BFS'
    }
}

def calculate_optimal_clusters(adjacency_matrix):
    """Calculate optimal number of clusters based on graph structure"""
    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz
    avg_degree = num_edges / num_nodes
    
    # For sparse graphs, we want larger clusters
    if avg_degree < 1:
        min_cluster_size = max(16, int(1/avg_degree))  # Ensure at least 16 nodes per cluster
    else:
        min_cluster_size = max(16, int(np.sqrt(num_nodes)))
    
    # Calculate number of clusters
    num_clusters = max(2, num_nodes // min_cluster_size)
    
    print(f"Graph stats:")
    print(f"- Nodes: {num_nodes}")
    print(f"- Edges: {num_edges}")
    print(f"- Avg degree: {avg_degree:.2f}")
    print(f"- Min cluster size: {min_cluster_size}")
    print(f"- Number of clusters: {num_clusters}")
    
    return num_clusters, min_cluster_size

def analyze_clustering_method(method_info, adjacency_matrix, feature_matrix, num_nodes):
    """Analyze clustering method with improved cluster size handling"""
    method_name = method_info['name']
    clustering_func = method_info['func']
    params = method_info['params'].copy()
    
    try:
        # Calculate optimal clustering parameters
        num_clusters, min_cluster_size = calculate_optimal_clusters(adjacency_matrix)
        
        # Set parameters based on graph structure
        if 'num_clusters' in params:
            params['num_clusters'] = num_clusters
        if 'max_edges_per_cluster' in params:
            params['max_edges_per_cluster'] = min_cluster_size * min_cluster_size
        
        if VERBOSE:
            print(f"\nRunning {method_name} with {num_clusters} clusters (min size: {min_cluster_size})")
        
        # Time decomposition phase
        decomp_start = time.perf_counter()
        clusters = clustering_func(adjacency_matrix, **params)
        decomp_time = time.perf_counter() - decomp_start
        
        # Merge small clusters
        merged_clusters = []
        current_cluster = []
        current_size = 0
        
        for cluster in sorted(clusters, key=len, reverse=True):
            if len(cluster) >= min_cluster_size:
                merged_clusters.append(cluster)
            else:
                if current_size + len(cluster) <= min_cluster_size * 2:
                    current_cluster.extend(cluster)
                    current_size += len(cluster)
                else:
                    if current_cluster:
                        merged_clusters.append(current_cluster)
                    current_cluster = cluster
                    current_size = len(cluster)
        
        if current_cluster:
            # Add any remaining nodes to the last cluster
            if merged_clusters:
                merged_clusters[-1].extend(current_cluster)
            else:
                merged_clusters.append(current_cluster)
        
        clusters = merged_clusters
        
        if VERBOSE:
            # Print cluster size statistics
            sizes = [len(c) for c in clusters]
            print(f"\nCluster statistics:")
            print(f"- Number of clusters: {len(clusters)}")
            print(f"- Average size: {np.mean(sizes):.1f}")
            print(f"- Min size: {np.min(sizes)}")
            print(f"- Max size: {np.max(sizes)}")
            
        # Prepare cluster data
        cluster_data = []
        for cluster_nodes in clusters:
            if method_name == 'pagerank':
                sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices_pagerank(
                    adjacency_matrix, feature_matrix, cluster_nodes)
            else:
                sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices(
                    adjacency_matrix, feature_matrix, cluster_nodes)
            # Check if sub_adj and sub_feat are valid
            if sub_adj is None or sub_feat is None or \
            sub_adj.shape[0] == 0 or sub_adj.shape[1] == 0 or \
            sub_feat.shape[0] == 0 or sub_feat.shape[1] == 0:
                print(f"Skipping cluster due to empty submatrices.")
                continue
            cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))

        if not cluster_data:
            raise RuntimeError("No valid clusters to process")
        
        # Time multiplication phase with memory-aware batching
        pipeline = Pipeline(batch_size=4)
        start_event = cuda.Event()
        end_event = cuda.Event()
        
        start_event.record()
        result_dict = pipeline.process_clusters(cluster_data, num_workers=1)  # Reduced workers
        end_event.record()
        end_event.synchronize()
        mult_time = start_event.time_till(end_event) / 1000.0  # Convert to seconds
        
        # Process results with proper unpacking
        result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
        node_counts = np.zeros(num_nodes, dtype=np.int32)
        cluster_times = []
        
        success = False
        for idx, data in result_dict.items():
            if data is not None:
                cluster_result, nodes_idx, timing = data  # Properly unpack three values
                result[nodes_idx] += cluster_result
                node_counts[nodes_idx] += 1
                cluster_times.append(timing)
                success = True

        # Average overlapping nodes
        mask = node_counts != 0
        result[mask] = result[mask] / node_counts[mask, np.newaxis]
        
        # Verify result
        is_correct = verify_result(result, adjacency_matrix, feature_matrix)

        if not is_correct:
            success = False
        
        cuda.Context.synchronize()  # Ensure GPU operations are complete
        
        return {
            'method': method_name,
            'decomp_time': decomp_time,
            'mult_time': mult_time,
            'num_clusters': len(clusters),
            'is_correct': is_correct,
            'processed_clusters': len([x for x in result_dict.values() if x is not None])
        }
        
        
    except Exception as e:
        print(f"Error in {method_name}: {str(e)}")
        cuda.Context.synchronize()  # Ensure GPU is in clean state
        return {
            'method': method_name,
            'decomp_time': float('nan'),
            'mult_time': float('nan'),
            'num_clusters': 0,
            'is_correct': False,
            'processed_clusters': 0,
            'error': str(e)
        }

def plot_clustering_comparison(results_df, output_dir='plots'):
    """Create comparison plots for decomposition and multiplication times"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Remove rows with NaN values
    results_df = results_df.dropna(subset=['decomp_time', 'mult_time'])
    
    # Plot decomposition times
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=results_df, x='graph_name', y='decomp_time', hue='method')
    plt.title('Decomposition Time Comparison')
    plt.xlabel('Graph')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decomp_times.png'))
    plt.close()
    
    # Plot multiplication times
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=results_df, x='graph_name', y='mult_time', hue='method')
    plt.title('Multiplication Time Comparison')
    plt.xlabel('Graph')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mult_times.png'))
    plt.close()
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    success_data = results_df.groupby(['method', 'graph_name'])['is_correct'].mean().reset_index()
    ax = sns.barplot(data=success_data, x='graph_name', y='is_correct', hue='method')
    plt.title('Success Rate by Method')
    plt.xlabel('Graph')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates.png'))
    plt.close()

if __name__ == '__main__':
    # Initialize results storage
    all_results = []
    
    try:
        context = cuda.Device(0).make_context()
        
        for graph_info in graphs:
            print(f"Testing graph {graph_info['name']}")
            
            # Prepare graph data
            adjacency_matrix = sp.lil_matrix((graph_info['num_nodes'], graph_info['num_nodes']), 
                                          dtype=np.float32)
            for node in graph_info['graph'].nodes:
                for neighbor in graph_info['graph'].neighbors(node):
                    adjacency_matrix[node, neighbor] = 1.0
            
            adjacency_matrix = adjacency_matrix.tocsr()
            feature_matrix = sp.csr_matrix(graph_info['feature_matrix'])
            
            # Test each clustering method
            for method_name, method_info in CLUSTERING_METHODS.items():
                try:
                    result = analyze_clustering_method(
                        method_info,
                        adjacency_matrix,
                        feature_matrix,
                        graph_info['num_nodes']
                    )
                    
                    # Add graph info to result
                    result.update({
                        'graph_name': graph_info['name'],
                        'graph_type': graph_info['type'],
                        'num_nodes': graph_info['num_nodes'],
                        'sparsity': graph_info['sparsity']
                    })
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {method_name} for {graph_info['name']}: {e}")
    
    finally:
        context.pop()
        context.detach()
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('clustering_comparison_results.csv', index=False)
    
    # Create plots
    plot_clustering_comparison(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = results_df.groupby('method').agg({
        'decomp_time': ['mean', 'std'],
        'mult_time': ['mean', 'std'],
        'is_correct': 'mean'
    })
    print(summary)
    
    # Additional error analysis
    error_summary = results_df[results_df['is_correct'] == False].groupby('method')['error'].value_counts()
    print("\nError Analysis:")
    print(error_summary)