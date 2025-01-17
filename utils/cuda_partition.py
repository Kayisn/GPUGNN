import random

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

QUEUE_SIZE = 16384  # Keep original size
MAX_BATCHES = 32  # Keep original size


def compute_fiedler_vector(adjacency_matrix):
    """Compute Fiedler vector for initial spectral partitioning"""
    laplacian = sp.csgraph.laplacian(adjacency_matrix, normed=True)
    try:
        _, vectors = eigsh(laplacian, k=2, which="SM")
        return vectors[:, 1]  # Fiedler vector
    except:
        print("Failed to compute Fiedler vector")
        return None


def estimate_graph_properties(adjacency_matrix, sample_size=1000):
    """Estimate graph properties using sampling"""
    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz

    # Sample nodes
    sample_nodes = random.sample(range(num_nodes), min(sample_size, num_nodes))

    # Analyze edge distribution
    degrees = np.diff(adjacency_matrix.indptr)
    sampled_degrees = degrees[sample_nodes]

    return {
        "avg_degree": num_edges / num_nodes,
        "density": num_edges / (num_nodes * num_nodes),
        "max_degree": np.max(degrees),
        "avg_sampled_degree": np.mean(sampled_degrees),
        "std_sampled_degree": np.std(sampled_degrees),
        "degree_skew": np.percentile(sampled_degrees, 90) / np.percentile(sampled_degrees, 50),
    }


def calculate_gpu_constraints():
    """Calculate GPU memory and compute constraints"""
    device = cuda.Device(0)
    attributes = device.get_attributes()

    return {
        "max_threads_block": attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
        "max_shared_memory": attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK],
        "num_sms": attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT],
        "warp_size": attributes[cuda.device_attribute.WARP_SIZE],
        "available_memory": cuda.mem_get_info()[0],
        "compute_capability": device.compute_capability(),
    }


def estimate_optimal_clusters(graph_props, gpu_constraints, feature_dim):
    """Estimate optimal cluster configuration based on GPU and graph properties"""

    # Calculate memory requirements per edge and feature
    bytes_per_edge = 4 * 3  # indices, data, weights
    bytes_per_feature = 4  # float32

    # Calculate maximum edges that can fit in GPU memory (use 80% of available)
    safe_memory = gpu_constraints["available_memory"] * 0.8
    max_edges_memory = safe_memory / bytes_per_edge

    # Consider feature matrix memory requirements
    feature_memory_per_node = feature_dim * bytes_per_feature

    # Adjust for GPU architecture
    threads_per_block = min(256, gpu_constraints["max_threads_block"])
    blocks_per_sm = 32  # Typical value, adjust based on occupancy calculation
    max_concurrent_threads = gpu_constraints["num_sms"] * blocks_per_sm * threads_per_block

    # Calculate target cluster sizes
    avg_cluster_edges = min(
        max_edges_memory / 10,  # Memory constraint
        max_concurrent_threads * graph_props["avg_degree"],  # Compute constraint
        graph_props["avg_degree"] * np.sqrt(graph_props["density"] * max_concurrent_threads),  # Density-aware sizing
    )

    # Estimate number of clusters
    num_clusters = max(2, int(graph_props["avg_degree"] * graph_props["density"] * gpu_constraints["num_sms"]))

    return {
        "target_edges_per_cluster": int(avg_cluster_edges),
        "num_clusters": num_clusters,
        "threads_per_block": threads_per_block,
        "max_concurrent_threads": max_concurrent_threads,
    }


def calculate_safe_cluster_limits(adjacency_matrix, feature_matrix, gpu_mem_info):
    """Calculate safe maximum cluster sizes based on GPU memory constraints"""
    free_mem, total_mem = gpu_mem_info
    safe_mem = free_mem * 0.8  # Use 80% of free memory

    # Memory requirements per edge and node
    bytes_per_edge = 4 * 3  # indices, data, weights (float32 + 2 * int32)
    bytes_per_node = 4  # cluster assignment (int32)

    # Feature matrix memory requirements
    feature_dim = feature_matrix.shape[1]
    bytes_per_feature_node = feature_dim * 4  # float32 per feature

    # Calculate memory needed for graph representation
    total_edges = adjacency_matrix.nnz
    total_nodes = adjacency_matrix.shape[0]

    # Reserve memory for basic graph structure
    base_memory = total_nodes * bytes_per_node + total_edges * bytes_per_edge

    remaining_mem = safe_mem - base_memory

    # Calculate maximum edges per cluster that fits in memory
    max_edges = int(remaining_mem / (bytes_per_edge + bytes_per_feature_node))
    max_edges_per_cluster = min(max_edges, int(total_edges * 0.2))  # No cluster should exceed 20% of total edges

    # Calculate number of clusters based on available memory
    min_clusters = total_edges // max_edges_per_cluster + 1
    max_clusters = min(
        total_nodes // 10, 32  # At least 10 nodes per cluster on average  # Maximum manageable number of clusters
    )

    num_clusters = max(min_clusters, min(max_clusters, int(np.sqrt(total_edges / max_edges_per_cluster))))

    return {
        "max_edges_per_cluster": max_edges_per_cluster,
        "num_clusters": num_clusters,
        "memory_per_cluster": max_edges_per_cluster * (bytes_per_edge + bytes_per_feature_node),
        "total_memory_needed": base_memory
        + num_clusters * max_edges_per_cluster * (bytes_per_edge + bytes_per_feature_node),
    }


def gpu_partition_graph(adjacency_matrix, kernel_manager=None, feature_matrix=None):
    """Edge-centric graph partitioning optimized for SpMM with pre-allocated memory"""
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()

    num_nodes = adjacency_matrix.shape[0]

    if feature_matrix is None:
        raise ValueError("Feature matrix is required for memory calculations")

    # Get GPU constraints and cluster limits
    gpu_mem_info = cuda.mem_get_info()
    cluster_limits = calculate_safe_cluster_limits(adjacency_matrix, feature_matrix, gpu_mem_info)

    max_edges_per_cluster = cluster_limits["max_edges_per_cluster"]
    num_clusters = cluster_limits["num_clusters"]

    print("\nCluster Configuration:")
    print(f"Max edges per cluster: {max_edges_per_cluster}")
    print(f"Number of clusters: {num_clusters}")
    print(f"Estimated memory per cluster: {cluster_limits['memory_per_cluster'] / 1024**2:.2f} MB")
    print(f"Total memory needed: {cluster_limits['total_memory_needed'] / 1024**2:.2f} MB")

    # Initialize arrays before GPU allocation
    assignments = np.full(num_nodes, -1, dtype=np.int32)
    edge_weights = np.zeros(adjacency_matrix.nnz, dtype=np.float32)
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)

    try:
        # Validate kernel manager and get kernels
        if not kernel_manager:
            raise RuntimeError("Kernel manager is required")

        compute_weights = kernel_manager.get_kernel("compute_edge_weights")
        bfs_kernel = kernel_manager.get_kernel("balanced_bfs")

        if not compute_weights or not bfs_kernel:
            raise RuntimeError(
                f"Required kernels not found: compute_weights={bool(compute_weights)}, bfs_kernel={bool(bfs_kernel)}"
            )

        # Calculate safe block and grid dimensions
        max_threads = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        threads_per_block = min(256, max_threads)
        num_blocks = (num_nodes + threads_per_block - 1) // threads_per_block

        block = (threads_per_block, 1, 1)
        grid = (num_blocks, 1)

        # Allocate and validate device memory
        try:
            d_row_ptr = cuda.mem_alloc(adjacency_matrix.indptr.nbytes)
            d_col_idx = cuda.mem_alloc(adjacency_matrix.indices.nbytes)
            d_assignments = cuda.mem_alloc(assignments.nbytes)
            d_sizes = cuda.mem_alloc(cluster_sizes.nbytes)
            d_edge_weights = cuda.mem_alloc(edge_weights.nbytes)

            # Initialize queue with proper alignment
            queue_struct_size = ((QUEUE_SIZE + MAX_BATCHES + 3) * 4 + 15) & ~15  # Ensure 16-byte alignment
            d_queue = cuda.mem_alloc(queue_struct_size)

            # Verify allocations
            if not all([d_row_ptr, d_col_idx, d_assignments, d_sizes, d_edge_weights, d_queue]):
                raise RuntimeError("Failed to allocate GPU memory")

        except cuda.MemoryError as e:
            raise RuntimeError(f"GPU memory allocation failed: {e}")

        # Copy data with verification
        try:
            cuda.memcpy_htod(d_row_ptr, adjacency_matrix.indptr.astype(np.int32))
            cuda.memcpy_htod(d_col_idx, adjacency_matrix.indices.astype(np.int32))
            cuda.memcpy_htod(d_assignments, assignments)
            cuda.memcpy_htod(d_sizes, cluster_sizes)
            cuda.memset_d32(d_edge_weights, 0, edge_weights.size)
        except cuda.Error as e:
            raise RuntimeError(f"Failed to copy data to GPU: {e}")

        # Launch compute_weights kernel with explicit synchronization
        try:
            compute_weights(d_row_ptr, d_col_idx, d_edge_weights, np.int32(num_nodes), block=block, grid=grid)
            cuda.Context.synchronize()
        except cuda.Error as e:
            raise RuntimeError(f"compute_weights kernel launch failed: {e}")

        # Get starting nodes based on degree
        degrees = np.diff(adjacency_matrix.indptr)
        start_nodes = (-degrees).argsort()

        # Process each cluster with proper synchronization
        bfs_kernel = kernel_manager.get_kernel("balanced_bfs")

        # Calculate proper block and grid dimensions for BFS kernel
        threads_per_block = 256  # Use standard CUDA block size
        block = (threads_per_block, 1, 1)
        grid = ((num_nodes + threads_per_block - 1) // threads_per_block, 1)

        print(f"Launch config - Grid: {grid}, Block: {block}, Nodes: {num_nodes}")

        # Process each cluster
        for cluster_id in range(num_clusters):
            # Find next unassigned high-degree node
            start_node = next((node for node in start_nodes if assignments[node] == -1), -1)
            if start_node == -1:
                break

            # Reset queue and update assignments
            cuda.memset_d32(d_queue, 0, queue_struct_size // 4)
            assignments[start_node] = cluster_id
            cuda.memcpy_htod(d_assignments, assignments)

            # Convert all arguments to proper types
            try:
                # Launch kernel directly
                bfs_kernel(
                    d_row_ptr,  # const int* row_ptr
                    d_col_idx,  # const int* col_idx
                    d_edge_weights,  # const float* edge_weights
                    d_assignments,  # int* cluster_assignments
                    d_sizes,  # unsigned int* cluster_sizes
                    np.int32(max_edges_per_cluster),  # const int max_edges_per_cluster
                    np.int32(num_nodes),  # const int num_nodes
                    np.int32(cluster_id),  # const int cluster_id
                    d_queue,  # BatchQueue* queue
                    block=block,
                    grid=grid,
                )

                cuda.Context.synchronize()

            except cuda.Error as e:
                print(f"BFS kernel launch failed for cluster {cluster_id}: {e}")
                print(f"Parameters: grid={grid}, block={block}")
                raise

            # Get updated assignments
            cuda.memcpy_dtoh(assignments, d_assignments)

        # Create clusters from assignments
        clusters = [[] for _ in range(num_clusters)]
        for node, cluster in enumerate(assignments):

            if cluster >= 0 and cluster < num_clusters:
                clusters[cluster].append(node)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        return clusters

    except cuda.Error as e:
        print(f"CUDA error: {str(e)}")
        raise
    finally:
        # Cleanup GPU memory
        locals_dict = locals()
        for name in ["d_row_ptr", "d_col_idx", "d_assignments", "d_sizes", "d_edge_weights", "d_queue"]:
            if name in locals_dict and locals_dict[name] is not None:
                try:
                    locals_dict[name].free()
                except cuda.Error:
                    pass
