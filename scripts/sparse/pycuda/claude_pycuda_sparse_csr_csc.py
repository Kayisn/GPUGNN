from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import allocate_gpu_memory, fetch_gpu_data, load_gpu_kernel


class SparseMatrixMultiplyCSCCSR:
    def __init__(self):
        self.kernel = next(load_gpu_kernel("sparse_csr_csc", "matmul"))

    def multiply(
        self,
        index: int,
        num_warmup: int,
        A: sp.csr_matrix,
        B: sp.csr_matrix,
        block_size: Tuple[int, int, int] = (32, 32, 1),
    ) -> np.ndarray:
        A_csr = A.tocsr().astype(np.float32)
        B_csc = B.tocsc().astype(np.float32)

        A_data, A_indices, A_indptr = A_csr.data, A_csr.indices, A_csr.indptr
        B_data, B_indices, B_indptr = B_csc.data, B_csc.indices, B_csc.indptr

        A_data_gpu = allocate_gpu_memory(A_data)
        A_indices_gpu = allocate_gpu_memory(A_indices)
        A_indptr_gpu = allocate_gpu_memory(A_indptr)
        B_data_gpu = allocate_gpu_memory(B_data)
        B_indices_gpu = allocate_gpu_memory(B_indices)
        B_indptr_gpu = allocate_gpu_memory(B_indptr)
        C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csc.shape[1] * A_data.dtype.itemsize)

        grid_size = (
            int(np.ceil(B_csc.shape[1] / block_size[0])),
            int(np.ceil(A_csr.shape[0] / block_size[1])),
            1,
        )

        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    self.kernel(
                        A_data_gpu,
                        A_indices_gpu,
                        A_indptr_gpu,
                        B_data_gpu,
                        B_indices_gpu,
                        B_indptr_gpu,
                        C_gpu,
                        np.int32(A_csr.shape[0]),
                        np.int32(A_csr.shape[1]),
                        np.int32(B_csc.shape[1]),
                        block=block_size,
                        grid=grid_size,
                    )
                    cuda.Context.synchronize()

            # Main
            with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                self.kernel(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    np.int32(A_csr.shape[0]),
                    np.int32(A_csr.shape[1]),
                    np.int32(B_csc.shape[1]),
                    block=block_size,
                    grid=grid_size,
                )
                cuda.Context.synchronize()

            C_dense = fetch_gpu_data(C_gpu, (A_csr.shape[0], B_csc.shape[1]), dtype=np.float32)
        finally:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()

        return C_dense


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.float32)

    smm_csr_csc = SparseMatrixMultiplyCSCCSR()
    return smm_csr_csc.multiply(index, num_warmup, adjacency_matrix, feature_matrix)
