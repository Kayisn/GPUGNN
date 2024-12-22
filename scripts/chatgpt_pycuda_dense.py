from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import load_gpu_func


# Define the PyCUDA-based multiplication method
def dense_matrix_multiply_pycuda(A, B, index, num_warmup):
    A_dense = A.toarray().astype(np.float32) if sp.issparse(A) else A.astype(np.float32)
    B_dense = B.astype(np.float32)

    A_gpu = cuda.mem_alloc(A_dense.nbytes)
    B_gpu = cuda.mem_alloc(B_dense.nbytes)
    C_gpu = cuda.mem_alloc(A_dense.shape[0] * B_dense.shape[1] * A_dense.dtype.itemsize)

    cuda.memcpy_htod(A_gpu, A_dense)
    cuda.memcpy_htod(B_gpu, B_dense)

    matmul = load_gpu_func("matmul")
    block_size = (16, 16, 1)
    grid_size = (
        int(np.ceil(B_dense.shape[1] / 16)),
        int(np.ceil(A_dense.shape[0] / 16)),
        1,
    )

    # Warmup
    with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
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

    # Main
    with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
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
    feature_matrix = graph_info["feature_matrix"]
    context = cuda.Device(0).make_context()

    # convert to dense matrix
    adjacency_matrix_dense = nx.to_numpy_array(graph, dtype=np.float32)
    feature_matrix_dense = feature_matrix.toarray()

    try:
        return dense_matrix_multiply_pycuda(adjacency_matrix_dense, feature_matrix_dense, index, num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
