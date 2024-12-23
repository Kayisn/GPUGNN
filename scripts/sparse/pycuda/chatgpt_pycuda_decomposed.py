import math
from pathlib import Path

import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import allocate_gpu_memory, fetch_gpu_data, load_gpu_kernel


class HierarchicalDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.clusters = []

    def decompose(self):
        # Simple decomposition - divide matrix into chunks
        n = self.matrix.shape[0]
        chunk_size = 1000  # Adjust based on available memory
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            self.clusters.append({"indices": list(range(i, end)), "size": end - i})

    def get_cluster_batch(self, available_memory):
        # Simple implementation - return all clusters
        return self.clusters


class SparseMatrixMultiplyDecomposed:
    def __init__(self):
        self.kernel = next(load_gpu_kernel("sparse_decomp", "matmul"))

    def multiply(self, index, num_warmup, A, B, block_size=(32, 32, 1)):
        A_csr = A.tocsr().astype(np.float32)
        B_csr = B.tocsr().astype(np.float32)

        A_data, A_indices, A_indptr = A_csr.data, A_csr.indices, A_csr.indptr
        B_data, B_indices, B_indptr = B_csr.data, B_csr.indices, B_csr.indptr

        A_data_gpu = allocate_gpu_memory(A_data)
        A_indices_gpu = allocate_gpu_memory(A_indices)
        A_indptr_gpu = allocate_gpu_memory(A_indptr)
        B_data_gpu = allocate_gpu_memory(B_data)
        B_indices_gpu = allocate_gpu_memory(B_indices)
        B_indptr_gpu = allocate_gpu_memory(B_indptr)
        C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

        grid_size = (
            int(np.ceil(B_csr.shape[1] / block_size[0])),
            int(np.ceil(A_csr.shape[0] / block_size[1])),
            1,
        )

        stream = cuda.Stream()

        C_dense = None
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
                        np.int32(B_csr.shape[1]),
                        block=block_size,
                        grid=grid_size,
                        stream=stream,
                    )
                    stream.synchronize()

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
                    np.int32(B_csr.shape[1]),
                    block=block_size,
                    grid=grid_size,
                    stream=stream,
                )
                stream.synchronize()

            C_dense = fetch_gpu_data(C_gpu, (A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        finally:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()

        return C_dense


def get_optimal_block_size():
    """Determine optimal block size based on GPU capabilities."""
    device = cuda.Device(0)
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)

    # Calculate optimal block dimension (multiple of warp size)
    block_dim = int(math.sqrt(max_threads_per_block))
    block_dim = (block_dim // warp_size) * warp_size
    return (block_dim, block_dim)


def estimate_memory_requirements(A, B):
    """Estimate GPU memory requirements for matrix multiplication."""
    memory_needed = (
        A.data.nbytes  # A_data
        + A.indices.nbytes  # A_indices
        + A.indptr.nbytes  # A_indptr
        + B.data.nbytes  # B_data
        + B.indices.nbytes  # B_indices
        + B.indptr.nbytes  # B_indptr
        + A.shape[0] * B.shape[1] * 4  # Result matrix (float32)
    )
    free_mem, _ = cuda.mem_get_info()
    return memory_needed, free_mem


def decompose_graph(adjacency_matrix, feature_matrix, max_size):
    """Basic graph decomposition for memory management."""
    num_rows = adjacency_matrix.shape[0]
    if num_rows <= max_size:
        return [(adjacency_matrix, feature_matrix)]

    num_parts = math.ceil(num_rows / max_size)
    subgraphs = []
    for i in range(num_parts):
        start_idx = i * max_size
        end_idx = min((i + 1) * max_size, num_rows)

        # Extract submatrices
        sub_adj = adjacency_matrix[start_idx:end_idx, :]
        sub_features = feature_matrix[start_idx:end_idx, :]

        subgraphs.append((sub_adj, sub_features))

    return subgraphs


def reorder_matrix_by_clusters(adjacency_matrix, feature_matrix, cluster_batch):
    """Reorder matrices according to clustering for better locality."""
    # Get new ordering from clusters
    new_order = []
    for cluster in cluster_batch:
        new_order.extend(cluster["indices"])

    # Create reverse mapping
    reverse_order = np.zeros_like(new_order)
    reverse_order[new_order] = np.arange(len(new_order))

    # Reorder matrices
    adj_reordered = adjacency_matrix[new_order][:, new_order]
    feat_reordered = feature_matrix[new_order]

    return adj_reordered, feat_reordered, reverse_order


def execute(graph_info, num_warmup=1):
    adjacency_matrix = nx.to_scipy_sparse_array(graph_info["graph"], format="lil", dtype=np.float32)

    # Create hierarchical decomposition
    decomp = HierarchicalDecomposition(adjacency_matrix)
    decomp.decompose()

    # Get cluster batch that fits in memory
    _, free_mem = cuda.mem_get_info()
    available_memory = free_mem * 0.8  # Leave 20% buffer
    cluster_batch = decomp.get_cluster_batch(available_memory)

    # Reorder matrices based on clustering
    adj_reordered, feat_reordered, reverse_order = reorder_matrix_by_clusters(
        adjacency_matrix, sp.csr_matrix(graph_info["feature_matrix"]), cluster_batch
    )

    smm_decomposed = SparseMatrixMultiplyDecomposed()
    return smm_decomposed.multiply(graph_info["index"], num_warmup, adj_reordered, feat_reordered)[reverse_order]
