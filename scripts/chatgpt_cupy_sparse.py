import cupy as cp
import numpy as np
import nvtx
import scipy.sparse as sp
from cupyx.scipy import sparse as cusp


# Define the CuPy sparse matrix multiplication method
def sparse_matrix_multiply_cusparse(index, adj_matrix, feat_matrix, num_warmup):
    with nvtx.annotate(f"prepare {index}", domain="chatgpt_cupy_sparse"):
        # Convert to CSR format and move to GPU
        adj_gpu = cusp.csr_matrix(adj_matrix)
        feat_gpu = cusp.csr_matrix(feat_matrix)

        # Ensure matrices are contiguous and optimally laid out
        adj_gpu.sort_indices()
        feat_gpu.sort_indices()

    # Warmup
    with nvtx.annotate(f"warmup {index}", domain="chatgpt_cupy_sparse"):
        for _ in range(num_warmup):
            result = adj_gpu.dot(feat_gpu)
            cp.cuda.stream.get_current_stream().synchronize()

    # Actual tests
    with nvtx.annotate(f"main {index}", domain="chatgpt_cupy_sparse"):
        result = adj_gpu.dot(feat_gpu)
        cp.cuda.stream.get_current_stream().synchronize()

    return result.get()


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    num_nodes = graph_info["num_nodes"]
    try:
        # Prepare adjacency matrix
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0
        adjacency_matrix = adjacency_matrix.tocsr()

        # Execute computation
        return sparse_matrix_multiply_cusparse(index, adjacency_matrix, feature_matrix, num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        cp.get_default_memory_pool().free_all_blocks()
