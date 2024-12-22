from pathlib import Path

import cupy as cp
import networkx as nx
import numpy as np
import nvtx


class DenseMatrixMultiplyCuPy:
    def __init__(self):
        pass

    def multiply(
        self,
        index: int,
        num_warmup: int,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Perform dense matrix multiplication using CuPy."""
        # Move to GPU
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)

        result = None
        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    cp.dot(A_gpu, B_gpu)
                    cp.cuda.stream.get_current_stream().synchronize()

            # Main
            with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                result = cp.dot(A_gpu, B_gpu).get()
                cp.cuda.stream.get_current_stream().synchronize()
        finally:
            cp.get_default_memory_pool().free_all_blocks()

        return result


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = np.array(graph_info["feature_matrix"], dtype=np.float32)
    adjacency_matrix = nx.to_numpy_array(graph, dtype=np.float32)

    dmm = DenseMatrixMultiplyCuPy()
    return dmm.multiply(index, num_warmup, adjacency_matrix, feature_matrix)