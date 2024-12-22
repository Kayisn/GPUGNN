import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp

from utils.cuda_helper import load_gpu_func


def gpu_partition_graph(adjacency_matrix, num_partitions):
    """Partition graph using GPU-accelerated local clustering"""
    # Convert to CSR if needed
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()

    num_nodes = adjacency_matrix.shape[0]

    try:
        # Validate inputs
        if num_nodes == 0:
            raise ValueError("Empty graph")
        if num_partitions <= 0:
            raise ValueError("Invalid number of partitions")

        # Calculate safe memory limits
        free_mem, total_mem = cuda.mem_get_info()
        max_allocation = min(free_mem * 0.8, 2**31 - 1)  # Stay within 32-bit indexing

        # Prepare arrays with size checks
        row_ptr = adjacency_matrix.indptr.astype(np.int32)
        col_idx = adjacency_matrix.indices.astype(np.int32)
        partition_labels = np.zeros(num_nodes, dtype=np.int32)

        # Verify allocation sizes
        total_alloc = row_ptr.nbytes + col_idx.nbytes + partition_labels.nbytes
        if total_alloc > max_allocation:
            raise cuda.Error("Required allocation exceeds safe memory limits")

        # Allocate GPU memory
        row_ptr_gpu = cuda.mem_alloc(row_ptr.nbytes)
        col_idx_gpu = cuda.mem_alloc(col_idx.nbytes)
        partition_labels_gpu = cuda.mem_alloc(partition_labels.nbytes)

        # Copy data to GPU
        cuda.memcpy_htod(row_ptr_gpu, row_ptr)
        cuda.memcpy_htod(col_idx_gpu, col_idx)

        # Compile and configure kernel
        kernel = load_gpu_func("partition_simplified")

        # Configure kernel parameters
        block_size = 256
        grid_size = (num_nodes + block_size - 1) // block_size
        max_edges_per_block = 1024  # Limit edges processed per node

        # Launch kernel with error checking
        try:
            kernel(
                row_ptr_gpu,
                col_idx_gpu,
                partition_labels_gpu,
                np.int32(num_nodes),
                np.int32(num_partitions),
                np.int32(max_edges_per_block),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
            )

            # Get results
            cuda.memcpy_dtoh(partition_labels, partition_labels_gpu)

        except cuda.Error as e:
            print(f"Kernel execution failed: {e}")
            raise

        finally:
            # Cleanup GPU memory
            row_ptr_gpu.free()
            col_idx_gpu.free()
            partition_labels_gpu.free()

        # Convert to clusters format
        clusters = [[] for _ in range(num_partitions)]
        for node, label in enumerate(partition_labels):
            clusters[min(label, num_partitions - 1)].append(node)

        # Remove empty clusters and ensure minimum size
        clusters = [c for c in clusters if len(c) >= 2]
        if not clusters:
            # Fallback: create single cluster
            clusters = [list(range(num_nodes))]

        return clusters

    except (cuda.Error, Exception) as e:
        print(f"GPU partitioning failed: {e}")
        # Fallback to simple sequential partitioning
        clusters = []
        nodes_per_cluster = max(2, num_nodes // num_partitions)
        for i in range(0, num_nodes, nodes_per_cluster):
            cluster = list(range(i, min(i + nodes_per_cluster, num_nodes)))
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters
