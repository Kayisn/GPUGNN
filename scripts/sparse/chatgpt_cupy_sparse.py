from pathlib import Path

import cupy as cp
import networkx as nx
import numpy as np
import nvtx
import scipy.sparse as sp
from cupyx.scipy import sparse as cusp


class SparseMatrixMultiplyCuPy:
    def __init__(self):
        pass

    def multiply(
        self,
        index: int,
        num_warmup: int,
        A: sp.csr_matrix,
        B: sp.csr_matrix,
    ) -> np.ndarray:
        """Perform sparse matrix multiplication using CuPy."""
        # Convert to CSR format and move to GPU
        A_gpu = cusp.csr_matrix(A)
        B_gpu = cusp.csr_matrix(B)

        # Ensure matrices are contiguous and optimally laid out
        A_gpu.sort_indices()
        B_gpu.sort_indices()

        result = None
        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    A_gpu.dot(B_gpu)
                    cp.cuda.stream.get_current_stream().synchronize()

            # Main
            with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                result = A_gpu.dot(B_gpu).get()
                cp.cuda.stream.get_current_stream().synchronize()
        finally:
            cp.get_default_memory_pool().free_all_blocks()

        return result


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.float32)

    smm = SparseMatrixMultiplyCuPy()
    return smm.multiply(index, num_warmup, adjacency_matrix, feature_matrix)
