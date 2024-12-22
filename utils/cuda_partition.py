import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import load_gpu_kernel


def gpu_partition_graph(adjacency_matrix, num_clusters, max_cluster_size=None):
    """
    Partition graph using edge-cut guided BFS clustering
    """
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()

    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz
    if max_cluster_size is None:
        max_cluster_size = min(num_nodes // num_clusters * 2, 4096)

    # Compile kernels
    cut_kernel, bfs_kernel = load_gpu_kernel("partition", "find_edge_cuts", "bfs_away_from_cuts")

    # Allocate GPU memory
    edge_weights = np.zeros(num_edges, dtype=np.float32)
    cut_markers = np.zeros(num_edges, dtype=np.int32)
    cluster_assignments = np.full(num_nodes, -1, dtype=np.int32)
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)

    gpu_arrays = {
        "row_ptr": cuda.mem_alloc(adjacency_matrix.indptr.astype(np.int32).nbytes),
        "col_idx": cuda.mem_alloc(adjacency_matrix.indices.astype(np.int32).nbytes),
        "edge_weights": cuda.mem_alloc(edge_weights.nbytes),
        "cut_markers": cuda.mem_alloc(cut_markers.nbytes),
        "cluster_assignments": cuda.mem_alloc(cluster_assignments.nbytes),
        "cluster_sizes": cuda.mem_alloc(cluster_sizes.nbytes),
    }

    # Copy data to GPU
    for name, arr in [
        ("row_ptr", adjacency_matrix.indptr.astype(np.int32)),
        ("col_idx", adjacency_matrix.indices.astype(np.int32)),
    ]:
        cuda.memcpy_htod(gpu_arrays[name], arr)

    # Find edge cuts
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size
    cut_threshold = 0.1  # Adjust this threshold based on your needs

    cut_kernel(
        gpu_arrays["row_ptr"],
        gpu_arrays["col_idx"],
        gpu_arrays["edge_weights"],
        gpu_arrays["cut_markers"],
        np.int32(num_nodes),
        np.float32(cut_threshold),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    # Get cut information back
    cuda.memcpy_dtoh(cut_markers, gpu_arrays["cut_markers"])

    # Find seed nodes near cuts
    cut_edges = np.where(cut_markers == 1)[0]
    seed_nodes = []
    for edge in cut_edges:
        # Find nodes on both sides of cut
        row_idx = np.searchsorted(adjacency_matrix.indptr[:-1], edge, side="right") - 1
        col_idx = adjacency_matrix.indices[edge]
        seed_nodes.extend([row_idx, col_idx])

    # Remove duplicates and limit to num_clusters seeds
    seed_nodes = list(set(seed_nodes))[:num_clusters]

    # Process clusters using BFS away from cuts
    frontier = np.zeros(max_cluster_size, dtype=np.int32)
    frontier_size = np.array([0], dtype=np.uint32)
    frontier_gpu = cuda.mem_alloc(frontier.nbytes)
    frontier_size_gpu = cuda.mem_alloc(frontier_size.nbytes)

    for cluster_id, seed in enumerate(seed_nodes):
        if cluster_assignments[seed] != -1:
            continue

        # Initialize frontier
        frontier[0] = seed
        frontier_size[0] = 1
        cluster_assignments[seed] = cluster_id

        cuda.memcpy_htod(frontier_gpu, frontier)
        cuda.memcpy_htod(frontier_size_gpu, frontier_size)
        cuda.memcpy_htod(gpu_arrays["cluster_assignments"], cluster_assignments)

        # Run BFS
        while frontier_size[0] > 0:
            bfs_kernel(
                gpu_arrays["row_ptr"],
                gpu_arrays["col_idx"],
                gpu_arrays["cut_markers"],
                gpu_arrays["cluster_assignments"],
                gpu_arrays["cluster_sizes"],
                frontier_gpu,
                frontier_size_gpu,
                np.uint32(max_cluster_size),
                np.int32(num_nodes),
                np.int32(cluster_id),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )

            cuda.memcpy_dtoh(frontier_size, frontier_size_gpu)

    # Get final assignments
    cuda.memcpy_dtoh(cluster_assignments, gpu_arrays["cluster_assignments"])

    # Handle unassigned nodes
    unassigned = np.where(cluster_assignments == -1)[0]
    for node in unassigned:
        # Assign to nearest cluster
        neighbors = adjacency_matrix[:, [node]].indices
        assigned_neighbors = [n for n in neighbors if cluster_assignments[n] >= 0]
        if assigned_neighbors:
            cluster_assignments[node] = cluster_assignments[assigned_neighbors[0]]
        else:
            cluster_assignments[node] = 0

    # Create final clusters
    clusters = [[] for _ in range(num_clusters)]
    for node, cluster in enumerate(cluster_assignments):
        if cluster >= 0 and cluster < num_clusters:
            clusters[cluster].append(node)

    # Clean up
    for arr in gpu_arrays.values():
        arr.free()
    frontier_gpu.free()
    frontier_size_gpu.free()

    return clusters
