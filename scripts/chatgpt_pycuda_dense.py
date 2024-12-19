import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule


# Define the PyCUDA-based multiplication method
def dense_matrix_multiply_pycuda(A, B, index, num_warmup):
    with nvtx.annotate(f"prepare {index}", domain="chatgpt_pycuda_dense"):
        A_dense = A.toarray().astype(np.float32) if sp.issparse(A) else A.astype(np.float32)
        B_dense = B.astype(np.float32)

        A_gpu = cuda.mem_alloc(A_dense.nbytes)
        B_gpu = cuda.mem_alloc(B_dense.nbytes)
        C_gpu = cuda.mem_alloc(A_dense.shape[0] * B_dense.shape[1] * A_dense.dtype.itemsize)

        cuda.memcpy_htod(A_gpu, A_dense)
        cuda.memcpy_htod(B_gpu, B_dense)

        mod = SourceModule(
            """
        __global__ void matmul(float *A, float *B, float *C, int widthA, int widthB) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < widthA && col < widthB) {
                float sum = 0;
                for (int k = 0; k < widthA; ++k) {
                    sum += A[row * widthA + k] * B[k * widthB + col];
                }
                C[row * widthB + col] = sum;
            }
        }
        """
        )

        matmul = mod.get_function("matmul")
        block_size = (16, 16, 1)
        grid_size = (
            int(np.ceil(B_dense.shape[1] / 16)),
            int(np.ceil(A_dense.shape[0] / 16)),
            1,
        )

    with nvtx.annotate(f"warmup {index}", domain="chatgpt_pycuda_dense"):
        for _ in range(num_warmup):
            matmul(
                A_gpu,
                B_gpu,
                C_gpu,
                np.int32(A_dense.shape[1]),
                np.int32(B_dense.shape[1]),
                block=block_size,
                grid=grid_size,
            )
            cuda.Context.synchronize()

    with nvtx.annotate(f"main {index}", domain="chatgpt_pycuda_dense"):
        matmul(
            A_gpu,
            B_gpu,
            C_gpu,
            np.int32(A_dense.shape[1]),
            np.int32(B_dense.shape[1]),
            block=block_size,
            grid=grid_size,
        )
        cuda.Context.synchronize()

    C_dense = np.empty((A_dense.shape[0], B_dense.shape[1]), dtype=np.float32)
    cuda.memcpy_dtoh(C_dense, C_gpu)

    A_gpu.free()
    B_gpu.free()
    C_gpu.free()

    return C_dense


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

        # convert to dense matrix
        feature_matrix_dense = feature_matrix.toarray()

        adjacency_matrix_dense = adjacency_matrix.toarray()

        return dense_matrix_multiply_pycuda(adjacency_matrix_dense, feature_matrix_dense, index, num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
