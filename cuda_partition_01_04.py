import numpy as np
import scipy.sparse as sp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

PARTITION_KERNELS = """
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_EDGES_PER_NODE 1024

__device__ float compute_local_density(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int node_id,
    const int num_nodes
) {
    __shared__ int shared_neighbors[BLOCK_SIZE];
    
    int start = row_ptr[node_id];
    int end = row_ptr[node_id + 1];
    int degree = end - start;
    
    if (degree == 0) return 0.0f;
    
    float density = 0.0f;
    int edge_count = 0;
    
    // Load node's direct neighbors
    for (int i = 0; i < min(degree, MAX_EDGES_PER_NODE); i++) {
        int neighbor = col_idx[start + i];
        int n_start = row_ptr[neighbor];
        int n_end = row_ptr[neighbor + 1];
        
        // Count edges between neighbors
        for (int j = n_start; j < n_end && j < n_start + MAX_EDGES_PER_NODE; j++) {
            for (int k = start; k < end && k < start + MAX_EDGES_PER_NODE; k++) {
                if (col_idx[j] == col_idx[k]) {
                    edge_count++;
                    break;
                }
            }
        }
    }
    
    int possible_edges = min(degree, MAX_EDGES_PER_NODE) * (min(degree, MAX_EDGES_PER_NODE) - 1) / 2;
    return possible_edges > 0 ? (float)edge_count / possible_edges : 0.0f;
}

__global__ void partition_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ partition_labels,
    unsigned int* __restrict__ partition_sizes,
    const int num_nodes,
    const int num_partitions,
    const int max_partition_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // Compute local density score
    float density = compute_local_density(row_ptr, col_idx, tid, num_nodes);
    
    // Map density to partition range
    int target_partition = (int)(density * (num_partitions - 0.01f));
    target_partition = min(target_partition, num_partitions - 1);
    
    // Try to assign to densest matching partition
    bool assigned = false;
    #pragma unroll
    for (int offset = 0; offset < num_partitions && !assigned; offset++) {
        int partition = (target_partition + offset) % num_partitions;
        unsigned int old_size = atomicAdd(&partition_sizes[partition], 1);
        
        if (old_size < max_partition_size) {
            partition_labels[tid] = partition;
            assigned = true;
        } else {
            atomicSub(&partition_sizes[partition], 1);
        }
    }
    
    // Fallback to least loaded partition if necessary
    if (!assigned) {
        unsigned int min_size = max_partition_size;
        int min_partition = 0;
        
        for (int p = 0; p < num_partitions; p++) {
            unsigned int size = partition_sizes[p];
            if (size < min_size) {
                min_size = size;
                min_partition = p;
            }
        }
        
        atomicAdd(&partition_sizes[min_partition], 1);
        partition_labels[tid] = min_partition;
    }
}
"""

def gpu_partition_graph(adjacency_matrix, num_clusters, feature_matrix=None, max_cluster_size=None):
    """Optimized graph partitioning using hybrid density-based clustering"""
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()
    
    num_nodes = adjacency_matrix.shape[0]
    if max_cluster_size is None:
        max_cluster_size = num_nodes // num_clusters * 2
    
    # Compile kernel
    mod = SourceModule(PARTITION_KERNELS)
    partition_kernel = mod.get_function("partition_kernel")
    
    # Prepare GPU arrays
    row_ptr = adjacency_matrix.indptr.astype(np.int32)
    col_idx = adjacency_matrix.indices.astype(np.int32)
    partition_labels = np.full(num_nodes, -1, dtype=np.int32)
    partition_sizes = np.zeros(num_clusters, dtype=np.uint32)
    
    # Allocate GPU memory
    d_row_ptr = cuda.mem_alloc(row_ptr.nbytes)
    d_col_idx = cuda.mem_alloc(col_idx.nbytes)
    d_partition_labels = cuda.mem_alloc(partition_labels.nbytes)
    d_partition_sizes = cuda.mem_alloc(partition_sizes.nbytes)
    
    try:
        # Copy data to GPU
        cuda.memcpy_htod(d_row_ptr, row_ptr)
        cuda.memcpy_htod(d_col_idx, col_idx)
        cuda.memcpy_htod(d_partition_labels, partition_labels)
        cuda.memcpy_htod(d_partition_sizes, partition_sizes)
        
        # Configure kernel launch
        block_size = min(256, num_nodes)
        grid_size = (num_nodes + block_size - 1) // block_size
        
        # Launch partitioning kernel
        partition_kernel(
            d_row_ptr,
            d_col_idx,
            d_partition_labels,
            d_partition_sizes,
            np.int32(num_nodes),
            np.int32(num_clusters),
            np.int32(max_cluster_size),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # Get results
        cuda.memcpy_dtoh(partition_labels, d_partition_labels)
        
        # Create and validate clusters
        clusters = [[] for _ in range(num_clusters)]
        for node, label in enumerate(partition_labels):
            if 0 <= label < num_clusters:
                clusters[label].append(node)
        
        # Filter out empty or tiny clusters
        valid_clusters = [c for c in clusters if len(c) >= 2]
        if not valid_clusters:
            raise RuntimeError("No valid clusters created")
        
        return valid_clusters
        
    except cuda.Error as e:
        print(f"CUDA Error in partitioning: {e}")
        raise
        
    finally:
        # Cleanup
        for arr in [d_row_ptr, d_col_idx, d_partition_labels, d_partition_sizes]:
            arr.free()
