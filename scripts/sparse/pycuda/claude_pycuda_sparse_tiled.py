from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import allocate_gpu_memory, fetch_gpu_data, load_gpu_kernel


class SparseMatrixMultiplyTiled:
    def __init__(self):
        self.kernel = next(load_gpu_kernel("sparse_tiled", "matmul"))

    def multiply(self, index, num_warmup, A, B, block_size=(32, 32, 1)):
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

        TILE_SIZE = 32
        block_size = (TILE_SIZE, TILE_SIZE, 1)
        grid_size = (
            (B_csc.shape[1] + TILE_SIZE - 1) // TILE_SIZE,
            (A_csr.shape[0] + TILE_SIZE - 1) // TILE_SIZE,
            1,
        )
        shared_mem_size = (TILE_SIZE * TILE_SIZE * 4 * 2) + (TILE_SIZE * TILE_SIZE * 4 * 2)

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
                        shared=shared_mem_size,
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
                    shared=shared_mem_size,
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
    smm_tiled = SparseMatrixMultiplyTiled()
    return smm_tiled.multiply(
        graph_info["index"],
        num_warmup,
        nx.to_scipy_sparse_array(graph_info["graph"], format="lil", dtype=np.float32),
        sp.csr_matrix(graph_info["feature_matrix"]),
    )
