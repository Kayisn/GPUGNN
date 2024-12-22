import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import torch

warnings.filterwarnings("ignore", ".*")  # Suppress warnings

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class DenseMatrixMultiplyPyTorch:
    def __init__(self):
        self.device = device

    def multiply(
        self,
        index: int,
        num_warmup: int,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Perform dense matrix multiplication using PyTorch."""
        # Convert matrices to PyTorch tensors
        A_tensor = torch.tensor(A, device=self.device)
        B_tensor = torch.tensor(B, device=self.device)

        result = None
        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    torch.mm(A_tensor, B_tensor)
                    torch.cuda.synchronize()

            # Main
            with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                result = torch.mm(A_tensor, B_tensor).cpu().numpy()
                torch.cuda.synchronize()
        finally:
            torch.cuda.empty_cache()

        return result


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = np.array(graph_info["feature_matrix"], dtype=np.float32)
    adjacency_matrix = nx.to_numpy_array(graph, dtype=np.float32)

    dmm = DenseMatrixMultiplyPyTorch()
    return dmm.multiply(index, num_warmup, adjacency_matrix, feature_matrix)
