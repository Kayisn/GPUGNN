import math

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule


class HierarchicalDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.clusters = []

    def decompose(self):
        # Simple decomposition - divide matrix into chunks
        n = self.matrix.shape[0]
        chunk_size = 1000  # Adjust based on available memory
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            self.clusters.append({"indices": list(range(i, end)), "size": end - i})

    def get_cluster_batch(self, available_memory):
        # Simple implementation - return all clusters
        return self.clusters


def get_optimal_block_size():
    """Determine optimal block size based on GPU capabilities."""
    device = cuda.Device(0)
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)

    # Calculate optimal block dimension (multiple of warp size)
    block_dim = int(math.sqrt(max_threads_per_block))
    block_dim = (block_dim // warp_size) * warp_size
    return (block_dim, block_dim)


def estimate_memory_requirements(A, B):
    """Estimate GPU memory requirements for matrix multiplication."""
    memory_needed = (
        A.data.nbytes  # A_data
        + A.indices.nbytes  # A_indices
        + A.indptr.nbytes  # A_indptr
        + B.data.nbytes  # B_data
        + B.indices.nbytes  # B_indices
        + B.indptr.nbytes  # B_indptr
        + A.shape[0] * B.shape[1] * 4  # Result matrix (float32)
    )
    free_mem, _ = cuda.mem_get_info()
    return memory_needed, free_mem


def decompose_graph(adjacency_matrix, feature_matrix, max_size):
    """Basic graph decomposition for memory management."""
    num_rows = adjacency_matrix.shape[0]
    if num_rows <= max_size:
        return [(adjacency_matrix, feature_matrix)]

    num_parts = math.ceil(num_rows / max_size)
    subgraphs = []
    for i in range(num_parts):
        start_idx = i * max_size
        end_idx = min((i + 1) * max_size, num_rows)

        # Extract submatrices
        sub_adj = adjacency_matrix[start_idx:end_idx, :]
        sub_features = feature_matrix[start_idx:end_idx, :]

        subgraphs.append((sub_adj, sub_features))

    return subgraphs


# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, index, num_warmup):
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

    # Create CUDA stream
    stream = cuda.Stream()

    try:
        with nvtx.annotate(f"prepare {index}", domain="chatgpt_pycuda_decomposed"):
            # Allocate GPU memory
            A_data_gpu = cuda.mem_alloc(A_data.nbytes)
            A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
            A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
            B_data_gpu = cuda.mem_alloc(B_data.nbytes)
            B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
            B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
            C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * np.float32().itemsize)

            # Safe memory transfer
            cuda.memcpy_htod(A_data_gpu, A_data)
            cuda.memcpy_htod(A_indices_gpu, A_indices)
            cuda.memcpy_htod(A_indptr_gpu, A_indptr)
            cuda.memcpy_htod(B_data_gpu, B_data)
            cuda.memcpy_htod(B_indices_gpu, B_indices)
            cuda.memcpy_htod(B_indptr_gpu, B_indptr)

            # Optimized CUDA kernel with binary search and shared memory
            mod = SourceModule(
                """
            __device__ inline int binary_search(const int* array, int left, int right, int target) {
                while (left <= right) {
                    int mid = (left + right) >> 1;
                    if (array[mid] == target) return mid;
                    if (array[mid] < target) left = mid + 1;
                    else right = mid - 1;
                }
                return -1;
            }

            __global__ void sparse_matmul(const float* __restrict__ A_data,
                                        const int* __restrict__ A_indices,
                                        const int* __restrict__ A_indptr,
                                        const float* __restrict__ B_data,
                                        const int* __restrict__ B_indices,
                                        const int* __restrict__ B_indptr,
                                        float* __restrict__ C,
                                        const int num_rows,
                                        const int num_cols,
                                        const int num_cols_B) {
                __shared__ float shared_sum[32][32];
                
                const int row = blockIdx.y * 32 + threadIdx.y;
                const int col = blockIdx.x * 32 + threadIdx.x;
                
                if (row < num_rows && col < num_cols_B) {
                    float sum = 0.0f;
                    const int row_start = A_indptr[row];
                    const int row_end = A_indptr[row + 1];
                    
                    #pragma unroll 4
                    for (int idx = row_start; idx < row_end; ++idx) {
                        const int k = A_indices[idx];
                        const float a_val = A_data[idx];
                        const int col_start = B_indptr[k];
                        const int col_end = B_indptr[k + 1];
                        
                        const int pos = binary_search(B_indices, col_start, col_end - 1, col);
                        if (pos != -1) {
                            sum += a_val * B_data[pos];
                        }
                    }
                    
                    shared_sum[threadIdx.y][threadIdx.x] = sum;
                    __syncthreads();
                    
                    if (threadIdx.x == 0) {
                        float final_sum = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            final_sum += shared_sum[threadIdx.y][i];
                        }
                        C[row * num_cols_B + col] = final_sum;
                    }
                }
            }
            """
            )

            sparse_matmul = mod.get_function("sparse_matmul")
            block_size = (32, 32, 1)  # Optimized block size
            grid_size = (
                int(np.ceil(B_csr.shape[1] / 32)),
                int(np.ceil(A_csr.shape[0] / 32)),
                1,
            )

        # Warmup
        with nvtx.annotate(f"warmup {index}", domain="chatgpt_pycuda_decomposed"):
            for _ in range(num_warmup):
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    np.int32(A_csr.shape[0]),
                    np.int32(A_csr.shape[1]),
                    np.int32(B_csr.shape[1]),
                    block=block_size,
                    grid=grid_size,
                    stream=stream,
                )
                stream.synchronize()

        # Main
        with nvtx.annotate(f"main {index}", domain="chatgpt_pycuda_decomposed"):
            sparse_matmul(
                A_data_gpu,
                A_indices_gpu,
                A_indptr_gpu,
                B_data_gpu,
                B_indices_gpu,
                B_indptr_gpu,
                C_gpu,
                np.int32(A_csr.shape[0]),
                np.int32(A_csr.shape[1]),
                np.int32(B_csr.shape[1]),
                block=block_size,
                grid=grid_size,
                stream=stream,
            )
            stream.synchronize()

        # Safe memory transfer back
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        return C_dense
    finally:
        # Ensure GPU memory is always freed
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


def reorder_matrix_by_clusters(adjacency_matrix, feature_matrix, cluster_batch):
    """Reorder matrices according to clustering for better locality."""
    # Get new ordering from clusters
    new_order = []
    for cluster in cluster_batch:
        new_order.extend(cluster["indices"])

    # Create reverse mapping
    reverse_order = np.zeros_like(new_order)
    reverse_order[new_order] = np.arange(len(new_order))

    # Reorder matrices
    adj_reordered = adjacency_matrix[new_order][:, new_order]
    feat_reordered = feature_matrix[new_order]

    return adj_reordered, feat_reordered, reverse_order


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    context = cuda.Device(0).make_context()

    # Create adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.float32)

    try:
        # Debug prints
        print(f"Matrix sizes - Adjacency: {adjacency_matrix.shape}, Features: {feature_matrix.shape}")
        print(f"Non-zero elements - Adjacency: {adjacency_matrix.nnz}, Features: {feature_matrix.nnz}")

        # Create hierarchical decomposition
        decomp = HierarchicalDecomposition(adjacency_matrix)
        decomp.decompose()

        # Get cluster batch that fits in memory
        _, free_mem = cuda.mem_get_info()
        available_memory = free_mem * 0.8  # Leave 20% buffer
        cluster_batch = decomp.get_cluster_batch(available_memory)

        # Reorder matrices based on clustering
        adj_reordered, feat_reordered, reverse_order = reorder_matrix_by_clusters(
            adjacency_matrix, feature_matrix, cluster_batch
        )

        # Restore original ordering
        return sparse_matrix_multiply_pycuda(adj_reordered, feat_reordered, index, num_warmup)[reverse_order]
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
