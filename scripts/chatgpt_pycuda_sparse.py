import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule


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

    with nvtx.annotate(f"prepare {index}", domain="chatgpt_pycuda_sparse"):
        # Allocate GPU memory for CSR components
        A_data_gpu = cuda.mem_alloc(A_data.nbytes)
        A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
        A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
        B_data_gpu = cuda.mem_alloc(B_data.nbytes)
        B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
        B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
        C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_data_gpu, A_data)
        cuda.memcpy_htod(A_indices_gpu, A_indices)
        cuda.memcpy_htod(A_indptr_gpu, A_indptr)
        cuda.memcpy_htod(B_data_gpu, B_data)
        cuda.memcpy_htod(B_indices_gpu, B_indices)
        cuda.memcpy_htod(B_indptr_gpu, B_indptr)

        # CUDA kernel for sparse matrix multiplication
        mod = SourceModule(
            """
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
        )

        sparse_matmul = mod.get_function("sparse_matmul")
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(B_csr.shape[1] / 16)),
            int(np.ceil(A_csr.shape[0] / 16)),
            1,
        )

    try:
        # Warmup runs
        with nvtx.annotate(f"warmup {index}", domain="chatgpt_pycuda_sparse"):
            for _ in range(num_warmup):
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    np.int32(A.shape[0]),
                    np.int32(A.shape[1]),
                    np.int32(B.shape[1]),
                    block=block_size,
                    grid=grid_size,
                )
                cuda.Context.synchronize()

        # Actual test runs
        with nvtx.annotate(f"main {index}", domain="chatgpt_pycuda_sparse"):
            sparse_matmul(
                A_data_gpu,
                A_indices_gpu,
                A_indptr_gpu,
                B_data_gpu,
                B_indices_gpu,
                B_indptr_gpu,
                C_gpu,
                np.int32(A.shape[0]),
                np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size,
            )
            cuda.Context.synchronize()

        # Copy the result back to host
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        return C_dense
    finally:
        # Free GPU memory
        A_data_gpu.free()
        A_indices_gpu.free()
        A_indptr_gpu.free()
        B_data_gpu.free()
        B_indices_gpu.free()
        B_indptr_gpu.free()
        C_gpu.free()


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    num_nodes = graph_info["num_nodes"]
    context = cuda.Device(0).make_context()
    try:
        # Perform multiplication (example using BFS and feature matrix)
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0
        adjacency_matrix = adjacency_matrix.tocsr()

        # Execute computation
        return sparse_matrix_multiply_pycuda(adjacency_matrix, feature_matrix, index, num_warmup=num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
