import warnings

import networkx as nx
import numpy as np
import nvtx
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*") # Suppress PyTorch warning

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def sparse_matrix_multiply_pytorch(adj_matrix, feat_matrix, index, num_warmup):
    # Convert scipy sparse matrix to PyTorch sparse tensor
    if not isinstance(feat_matrix, sp.csr_matrix):
        feat_matrix = feat_matrix.to_sparse_csr()
        feat_matrix = sp.csr_matrix(
            (
                feat_matrix.values().cpu().numpy(),
                feat_matrix.indices().cpu().numpy(),
                feat_matrix.crow_indices().cpu().numpy(),
            ),
            shape=feat_matrix.shape,
        )

    with nvtx.annotate(f"prepare {index}", domain="chatgpt_pytorch_sparse"):
        # Convert adjacency matrix to CSR format
        adj_matrix = adj_matrix.tocsr()

        # Convert to PyTorch sparse tensors
        adj_sparse = torch.sparse_csr_tensor(
            torch.from_numpy(adj_matrix.indptr).to(device),
            torch.from_numpy(adj_matrix.indices).to(device),
            torch.from_numpy(adj_matrix.data).to(device),
            size=adj_matrix.shape,
        )

        feat_sparse = torch.sparse_csr_tensor(
            torch.from_numpy(feat_matrix.indptr).to(device),
            torch.from_numpy(feat_matrix.indices).to(device),
            torch.from_numpy(feat_matrix.data).to(device),
            size=feat_matrix.shape,
        )

    # Warmup
    with nvtx.annotate(f"warmup {index}", domain="chatgpt_pytorch_sparse"):
        for _ in range(num_warmup):
            result = torch.sparse.mm(adj_sparse, feat_sparse)
            torch.cuda.synchronize()

    # Main
    with nvtx.annotate(f"main {index}", domain="chatgpt_pytorch_sparse"):
        result = torch.sparse.mm(adj_sparse, feat_sparse)
        torch.cuda.synchronize()

    return result.to_dense().cpu().numpy()


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    # Prepare adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.float32)

    try:
        return sparse_matrix_multiply_pytorch(adjacency_matrix, feature_matrix, index, num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        torch.cuda.empty_cache()
