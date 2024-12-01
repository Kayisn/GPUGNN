import numpy as np
import nvtx
import scipy.sparse as sp
import torch

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def sparse_matrix_multiply_pytorch(adj_matrix, feat_matrix, index, num_warmup=2, num_runs=5):
    # Convert scipy sparse matrix to PyTorch sparse tensor
    if not isinstance(adj_matrix, sp.csr_matrix):
        adj_matrix = adj_matrix.to_sparse_csr()
        adj_matrix = sp.csr_matrix(
            (
                adj_matrix.values().cpu().numpy(),
                adj_matrix.indices().cpu().numpy(),
                adj_matrix.crow_indices().cpu().numpy(),
            ),
            shape=adj_matrix.shape,
        )

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

    # Actual runs
    with nvtx.annotate(f"main {index}", domain="chatgpt_pytorch_sparse"):
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = torch.sparse.mm(adj_sparse, feat_sparse)
            end_event.record()
            torch.cuda.synchronize()

            times.append(start_event.elapsed_time(end_event))

    return result, np.mean(times), np.std(times)


def execute(graph_info, num_warmup=1, num_runs=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    num_nodes = graph_info["num_nodes"]

    # Prepare adjacency matrix
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    adjacency_matrix = adjacency_matrix.tocsr()

    try:
        return sparse_matrix_multiply_pytorch(
            adjacency_matrix, feature_matrix, index, num_warmup=num_warmup, num_runs=num_runs
        )
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        # Clear cache
        torch.cuda.empty_cache()
