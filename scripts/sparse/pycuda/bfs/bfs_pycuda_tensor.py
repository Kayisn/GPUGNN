import multiprocessing
import threading
from pathlib import Path
from queue import Queue

import networkit as nk
import networkx as nx
import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from numpy import random as np_random

from utils.cuda_helper import load_gpu_kernel
from utils.cuda_partition import gpu_partition_graph


def get_gpu_capabilities():
    """Check GPU capabilities including tensor core support"""
    device = cuda.Device(0)

    # Check for Tensor Core support (SM 7.0 or higher)
    compute_capability = device.compute_capability()
    has_tensor_cores = compute_capability[0] >= 7

    return {
        "has_tensor_cores": has_tensor_cores,
        "compute_capability": compute_capability,
        "total_memory": device.total_memory(),
    }


def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)


# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, index, num_warmup):
    # Ensure A and B are in CSR format
    A_csr = A.tocsr().astype(np.float32)
    B_csr = B.tocsr().astype(np.float32)

    # Extract CSR components for A
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr

    # Extract CSC components for B
    B_data = B_csr.data
    B_indices = B_csr.indices
    B_indptr = B_csr.indptr

    try:
        # Choose appropriate kernel based on GPU capabilities
        # if get_gpu_capabilities()["has_tensor_cores"]:
        if False:
            print("Using tensor core kernel for sparse matrix multiplication.")
            kernel_name = "sparse_tensor"
        else:
            print("Using standard kernel for sparse matrix multiplication.")
            kernel_name = "sparse"

        # Compile the selected kernel
        sparse_matmul = next(load_gpu_kernel(kernel_name, "matmul"))

        # TODO: Adjust block size for tensor cores if available
        block_size = (16, 16, 1)

        # Ensure A and B have valid shapes
        if A.shape[0] == 0 or A.shape[1] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
            raise ValueError("Input matrices A and B must have non-zero dimensions.")

        # Log the sizes of the CSR components
        print(f"A_data size: {A_data.nbytes}, A_indices size: {A_indices.nbytes}, A_indptr size: {A_indptr.nbytes}")
        print(f"B_data size: {B_data.nbytes}, B_indices size: {B_indices.nbytes}, B_indptr size: {B_indptr.nbytes}")

        # Check if A_data is empty
        if A_data.nbytes == 0:
            raise ValueError("Matrix A is empty, skipping this cluster.")

        # Allocate GPU memory for CSR components
        A_data_gpu = cuda.mem_alloc(A_data.nbytes)
        A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
        A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
        B_data_gpu = cuda.mem_alloc(B_data.nbytes)
        B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
        B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
        C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

        # Copy data to GPU (synchronous)
        cuda.memcpy_htod(A_data_gpu, A_data)
        cuda.memcpy_htod(A_indices_gpu, A_indices)
        cuda.memcpy_htod(A_indptr_gpu, A_indptr)
        cuda.memcpy_htod(B_data_gpu, B_data)
        cuda.memcpy_htod(B_indices_gpu, B_indices)
        cuda.memcpy_htod(B_indptr_gpu, B_indptr)

        grid_size = (
            int(np.ceil(B_csr.shape[1] / 16)),
            int(np.ceil(A_csr.shape[0] / 16)),
            1,
        )

        # Warmup
        with nvtx.annotate(f"warmup {index}", domain=Path(__file__).stem):
            for _ in range(num_warmup):
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    np.int32(A.shape[0]),
                    np.int32(A.shape[1]),
                    np.int32(B.shape[1]),
                    block=block_size,
                    grid=grid_size,
                )
                cuda.Context.synchronize()

        # Main
        with nvtx.annotate(f"main {index}", domain=Path(__file__).stem):
            sparse_matmul(
                A_data_gpu,
                A_indices_gpu,
                A_indptr_gpu,
                B_data_gpu,
                B_indices_gpu,
                B_indptr_gpu,
                C_gpu,
                np.int32(A.shape[0]),
                np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size,
            )
            cuda.Context.synchronize()

        # Synchronous copy back
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        # Free GPU memory
        A_data_gpu.free()
        A_indices_gpu.free()
        A_indptr_gpu.free()
        B_data_gpu.free()
        B_indices_gpu.free()
        B_indptr_gpu.free()
        C_gpu.free()

        return C_dense

    finally:
        # Cleanup on error
        try:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()
        except:
            pass


def extract_submatrices(adjacency_matrix, feature_matrix, cluster_nodes):
    """Extract submatrices including all necessary connections"""
    nodes_idx = sorted(cluster_nodes)

    # Get all nodes that are connected to the cluster nodes
    connected_nodes = set()
    for node in nodes_idx:
        row = adjacency_matrix[:, [node]].tocsr()
        connected_nodes.update(row.indices)

    # Include both cluster nodes and their neighbors
    all_nodes = sorted(set(nodes_idx) | connected_nodes)

    # Extract the relevant submatrices
    sub_adj = adjacency_matrix[nodes_idx, :][:, all_nodes]
    sub_feat = feature_matrix[all_nodes, :]

    if not nodes_idx or not all_nodes:
        return None, None, None, None

    # Ensure sub_adj and sub_feat are in CSR format
    sub_adj = sub_adj.tocsr()
    sub_feat = sub_feat.tocsr()
    return sub_adj, sub_feat, nodes_idx, all_nodes


def process_cluster_worker(args):
    idx, data = args
    try:
        # Each process creates its own context
        context = cuda.Device(0).make_context()
        try:
            sub_adj, sub_feat, nodes_idx, all_nodes, index, num_warmup = data

            # Check if sub_adj and sub_feat have valid shapes
            if sub_adj.shape[0] == 0 or sub_adj.shape[1] == 0 or sub_feat.shape[0] == 0 or sub_feat.shape[1] == 0:
                print(f"Skipping cluster {idx} due to empty submatrices.")
                return idx, None, None

            result = sparse_matrix_multiply_pycuda(sub_adj, sub_feat, index, num_warmup)
            return idx, result[: len(nodes_idx)], nodes_idx
        finally:
            context.pop()
    except Exception as e:
        print(f"Error processing cluster {idx}: {e}")
        return idx, None, None


class Pipeline:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.lock = threading.Lock()

    def process_batch(self, batch_idx, batch_data, index, num_warmup):
        context = cuda.Device(0).make_context()  # Create a CUDA context for this thread
        try:
            results = []
            for idx, data in batch_data:
                try:
                    sub_adj, sub_feat, nodes_idx, all_nodes = data
                    result, timing, _ = sparse_matrix_multiply_pycuda(sub_adj, sub_feat, index, num_warmup)
                    results.append((idx, result[: len(nodes_idx)], nodes_idx, timing))
                except Exception as e:
                    print(f"Error in batch {batch_idx}, cluster {idx}: {e}")
                    continue

            with self.lock:
                self.output_queue.put((batch_idx, results))
        finally:
            context.pop()  # Ensure the context is popped after processing

    def process_clusters(self, cluster_data, num_workers=2):
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_cluster_worker, enumerate(cluster_data))
        result_dict = {}
        for idx, cluster_result, nodes_idx in results:
            if cluster_result is not None:
                result_dict[idx] = (cluster_result, nodes_idx)
            else:
                print(f"Cluster {idx} was skipped or failed; excluding from results.")
                result_dict[idx] = None
        return result_dict


def calculate_pagerank_gpu(adjacency_matrix, damping=0.85, max_iterations=100, tolerance=1e-6):
    """Calculate PageRank scores using GPU acceleration"""
    num_nodes = adjacency_matrix.shape[0]

    # Convert to CSR format if needed
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()

    # Prepare GPU memory
    rank_current = np.ones(num_nodes, dtype=np.float32) / num_nodes
    rank_next = np.zeros_like(rank_current)

    # Allocate GPU memory
    rank_current_gpu = cuda.mem_alloc(rank_current.nbytes)
    rank_next_gpu = cuda.mem_alloc(rank_next.nbytes)
    row_ptr_gpu = cuda.mem_alloc(adjacency_matrix.indptr.nbytes)
    col_idx_gpu = cuda.mem_alloc(adjacency_matrix.indices.nbytes)
    values_gpu = cuda.mem_alloc(adjacency_matrix.data.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(rank_current_gpu, rank_current)
    cuda.memcpy_htod(row_ptr_gpu, adjacency_matrix.indptr)
    cuda.memcpy_htod(col_idx_gpu, adjacency_matrix.indices)
    cuda.memcpy_htod(values_gpu, adjacency_matrix.data)

    # Compile kernel
    pagerank_kernel = next(load_gpu_kernel("pagerank_iteration", "pagerank_iteration"))

    # Set up grid and block dimensions
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size

    # PageRank iteration
    for _ in range(max_iterations):
        pagerank_kernel(
            rank_current_gpu,
            rank_next_gpu,
            row_ptr_gpu,
            col_idx_gpu,
            values_gpu,
            np.float32(damping),
            np.int32(num_nodes),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        # Swap buffers
        rank_current_gpu, rank_next_gpu = rank_next_gpu, rank_current_gpu

        # Check convergence (optional)
        cuda.memcpy_dtoh(rank_next, rank_next_gpu)
        if np.abs(rank_next - rank_current).max() < tolerance:
            break

        rank_current = rank_next.copy()

    # Get final results
    cuda.memcpy_dtoh(rank_current, rank_current_gpu)

    # Clean up
    rank_current_gpu.free()
    rank_next_gpu.free()
    row_ptr_gpu.free()
    col_idx_gpu.free()
    values_gpu.free()

    return rank_current


def bfs_cluster(adjacency_matrix, start_node, max_edges):
    """Create a cluster using BFS until reaching max_edges"""
    nodes_in_cluster = set()
    edges_in_cluster = 0
    queue = [start_node]
    nodes_in_cluster.add(start_node)

    while queue and edges_in_cluster < max_edges:
        current = queue.pop(0)
        row = adjacency_matrix[current].tocsr()
        neighbors = row.indices

        for neighbor in neighbors:
            if neighbor not in nodes_in_cluster:
                nodes_in_cluster.add(neighbor)
                queue.append(neighbor)
            edges_in_cluster += 1  # Count all edges, even to existing nodes

    return list(nodes_in_cluster)


def create_clusters_pagerank(adjacency_matrix, max_edges_per_cluster, seed=None):
    """Create clusters using PageRank-based node selection"""
    rng = np_random.RandomState(seed)  # Create a random number generator

    num_nodes = adjacency_matrix.shape[0]
    remaining_edges = set((i, j) for i, j in zip(*adjacency_matrix.nonzero()))
    unvisited = set(range(num_nodes))
    clusters = []

    # Calculate PageRank scores
    pagerank_scores = calculate_pagerank_gpu(adjacency_matrix)

    while unvisited:
        # Normalize scores for unvisited nodes
        valid_scores = np.zeros(num_nodes)
        for node in unvisited:
            valid_scores[node] = pagerank_scores[node]

        if valid_scores.sum() == 0:
            start_node = min(unvisited)  # Fallback to deterministic selection
        else:
            # Normalize to probabilities
            valid_scores = valid_scores / valid_scores.sum()
            start_node = rng.choice(num_nodes, p=valid_scores)  # Use rng instead of np_random

        # Rest of the function remains the same
        cluster_nodes = bfs_cluster(adjacency_matrix, start_node, max_edges_per_cluster)
        clusters.append(cluster_nodes)

        for node in cluster_nodes:
            row = adjacency_matrix[node].tocsr()
            for neighbor in row.indices:
                remaining_edges.discard((node, neighbor))

        unvisited -= set(cluster_nodes)

    # Handle remaining edges
    while remaining_edges:
        edge = remaining_edges.pop()
        added = False
        for cluster in clusters:
            if edge[0] in cluster or edge[1] in cluster:
                cluster.extend([edge[0], edge[1]])
                cluster = list(set(cluster))
                added = True
                break
        if not added:
            clusters.append([edge[0], edge[1]])

    return clusters


def create_clusters_metis(adjacency_matrix, num_clusters):
    """
    Create clusters using METIS-style graph partitioning via NetworKit.
    Runtime complexity: O(|E|) - near linear in the number of edges
    """
    # Convert scipy sparse matrix to NetworkX graph using the correct function
    if hasattr(nx, "from_scipy_sparse_matrix"):  # older versions
        nx_graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    else:  # newer versions
        nx_graph = nx.from_scipy_sparse_array(adjacency_matrix)

    # Convert NetworkX graph to NetworKit graph
    g = nk.nxadapter.nx2nk(nx_graph)

    # Run partitioning with PLM (Parallel Louvain Method)
    communities = nk.community.PLM(g).run().getPartition()

    # Convert partition to clusters
    num_communities = communities.numberOfSubsets()
    clusters = [[] for _ in range(num_communities)]
    for node in range(g.numberOfNodes()):
        comm_id = communities[node]
        clusters[comm_id].append(node)

    # Remove empty clusters and ensure balanced sizes
    clusters = [c for c in clusters if c]

    # Ensure we don't exceed desired number of clusters
    if len(clusters) > num_clusters:
        # Merge smallest clusters
        clusters.sort(key=len)
        while len(clusters) > num_clusters:
            smallest = clusters.pop(0)
            clusters[0].extend(smallest)

    return clusters


def create_clusters_metis(adjacency_matrix, num_clusters):
    """
    Create clusters using METIS-style graph partitioning via NetworKit.
    Runtime complexity: O(|E|) - near linear in the number of edges
    """
    # Convert to NetworkX then NetworKit format
    nx_graph = nx.from_scipy_sparse_array(adjacency_matrix)
    g = nk.nxadapter.nx2nk(nx_graph)

    # Run partitioning
    communities = nk.community.PLM(g).run().getPartition()

    # Convert results to cluster format
    clusters = [[] for _ in range(communities.numberOfSubsets())]
    for node in range(g.numberOfNodes()):
        comm_id = communities[node]
        clusters[comm_id].append(node)

    # Balance cluster sizes
    clusters = [c for c in clusters if c]
    if len(clusters) > num_clusters:
        clusters.sort(key=len)
        while len(clusters) > num_clusters:
            smallest = clusters.pop(0)
            clusters[0].extend(smallest)

    return clusters


def create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters):
    """
    Create clusters using GPU-accelerated METIS/BFS hybrid partitioning
    """
    return gpu_partition_graph(adjacency_matrix, num_clusters)


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    num_nodes = graph_info["num_nodes"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    context = cuda.Device(0).make_context()

    # Prepare adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=np.float32)

    try:
        # Calculate batch size dynamically based on GPU SMs
        num_sms = get_num_sms()
        threads_per_sm = 1024  # Adjust based on your GPU architecture
        total_threads = num_sms * threads_per_sm
        threads_per_edge = 1  # Define based on kernel requirements
        batch_size = total_threads // threads_per_edge

        # decomposition phase
        with nvtx.annotate(f"decomp {index}", domain=Path(__file__).stem):
            # Define the number of clusters - more conservative estimate
            cluster_size = max(int(np.sqrt(num_nodes)), batch_size)  # Increased minimum size
            num_clusters = max(2, num_nodes // cluster_size)

            # Create clusters using METIS-style partitioning
            clusters = create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters)

            # Prepare cluster data
            cluster_data = []
            for cluster_nodes in clusters:
                sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices(
                    adjacency_matrix, feature_matrix, cluster_nodes
                )
                # Check if sub_adj and sub_feat are valid
                if (
                    sub_adj is None
                    or sub_feat is None
                    or sub_adj.shape[0] == 0
                    or sub_adj.shape[1] == 0
                    or sub_feat.shape[0] == 0
                    or sub_feat.shape[1] == 0
                ):
                    print(f"Skipping cluster due to empty submatrices.")
                    continue
                cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes, index, num_warmup))

        # Process clusters in parallel
        pipeline = Pipeline(batch_size=4)
        result_dict = pipeline.process_clusters(cluster_data, num_workers=2)

        # Combine results with proper error handling
        result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
        node_counts = np.zeros(num_nodes, dtype=np.int32)
        success = False
        for idx in range(len(cluster_data)):
            if idx not in result_dict or result_dict[idx] is None:
                continue

            cluster_result, nodes_idx = result_dict[idx]
            if cluster_result is None:
                continue  # Skip failed clusters

            result[nodes_idx] += cluster_result
            node_counts[nodes_idx] += 1
            success = True

        if not success:
            raise RuntimeError("All clusters failed to process")

        # Average results for overlapping nodes
        mask = node_counts != 0
        result[mask] = result[mask] / node_counts[mask, np.newaxis]

        return result
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
