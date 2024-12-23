import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")  # Suppress PyTorch warning

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class SparseMatrixMultiplyPyTorch:
    def __init__(self):
        self.device = device

    def multiply(
        self,
        index: int,
        num_warmup: int,
        A: sp.csr_matrix,
        B: sp.csr_matrix,
    ) -> np.ndarray:
        """Perform sparse matrix multiplication using PyTorch."""
        # Convert matrices to CSR format
        A_csr = A.tocsr()
        B_csr = B.tocsr()

        # Convert to PyTorch sparse tensors
        A_sparse = torch.sparse_csr_tensor(
            torch.from_numpy(A_csr.indptr).to(self.device),
            torch.from_numpy(A_csr.indices).to(self.device),
            torch.from_numpy(A_csr.data).to(self.device),
            size=A_csr.shape,
        )

        B_sparse = torch.sparse_csr_tensor(
            torch.from_numpy(B_csr.indptr).to(self.device),
            torch.from_numpy(B_csr.indices).to(self.device),
            torch.from_numpy(B_csr.data).to(self.device),
            size=B_csr.shape,
        )

        result = None
        try:
            # Warmup
            with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
                for _ in range(num_warmup):
                    torch.sparse.mm(A_sparse, B_sparse)
                    torch.cuda.synchronize()

            # Main
            with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
                result = torch.sparse.mm(A_sparse, B_sparse).to_dense().cpu().numpy()
                torch.cuda.synchronize()
        finally:
            torch.cuda.empty_cache()

        return result


def execute(graph_info, num_warmup=1):
    smm = SparseMatrixMultiplyPyTorch()
    return smm.multiply(
        graph_info["index"],
        num_warmup,
        nx.to_scipy_sparse_array(graph_info["graph"], format="lil", dtype=np.float32),
        sp.csr_matrix(graph_info["feature_matrix"]),
    )
