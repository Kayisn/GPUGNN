from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import allocate_gpu_memory, fetch_gpu_data, load_gpu_kernel


class DenseMatrixMultiply:
    def __init__(self):
        self.kernel = next(load_gpu_kernel("dense", "matmul"))

    def multiply(self, index, num_warmup, A, B, block_size=(16, 16, 1)):
        A_dense = A.toarray().astype(np.float32) if sp.issparse(A) else A.astype(np.float32)
        B_dense = B.astype(np.float32)

        A_gpu = allocate_gpu_memory(A_dense)
        B_gpu = allocate_gpu_memory(B_dense)
        C_gpu = cuda.mem_alloc(A_dense.shape[0] * B_dense.shape[1] * A_dense.dtype.itemsize)

        grid_size = (
            int(np.ceil(B_dense.shape[1] / block_size[0])),
            int(np.ceil(A_dense.shape[0] / block_size[1])),
            1,
        )

        C_dense = None
        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    self.kernel(
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
                self.kernel(
                    A_gpu,
                    B_gpu,
                    C_gpu,
                    np.int32(A_dense.shape[1]),
                    np.int32(B_dense.shape[1]),
                    block=block_size,
                    grid=grid_size,
                )
                cuda.Context.synchronize()

            C_dense = fetch_gpu_data(C_gpu, (A_dense.shape[0], B_dense.shape[1]), dtype=np.float32)
        finally:
            A_gpu.free()
            B_gpu.free()
            C_gpu.free()

        return C_dense


def execute(graph_info, num_warmup=1):
    dmm = DenseMatrixMultiply()

    # Convert feature matrix to dense if it is sparse
    feature_matrix = graph_info["feature_matrix"]
    if hasattr(feature_matrix, "todense"):
        feature_matrix = feature_matrix.todense()
        
    return dmm.multiply(
        graph_info["index"],
        num_warmup,
        nx.to_numpy_array(graph_info["graph"], dtype=np.float32),
        sp.csr_matrix(graph_info["feature_matrix"]).toarray(),
    )
