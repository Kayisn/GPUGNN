import numpy as np
import scipy.sparse as sp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

PARTITION_KERNELS = """
#define BLOCK_SIZE 256
#define MAX_QUEUE_SIZE 1024
#define WARP_SIZE 32

__device__ float compute_density(
    const int* row_ptr,
    const int* col_idx,
    int node,
    int start,
    int end
) {
    int edge_count = 0;
    int possible_edges = (end - start) * (end - start - 1) / 2;
    
    for (int i = start; i < end; i++) {
        int neighbor1 = col_idx[i];
        int n1_start = row_ptr[neighbor1];
        int n1_end = row_ptr[neighbor1 + 1];
        
        for (int j = n1_start; j < n1_end; j++) {
            for (int k = start; k < end; k++) {
                if (col_idx[j] == col_idx[k]) {
                    edge_count++;
                    break;
                }
            }
        }
    }
    
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
    __shared__ int shared_edges[MAX_QUEUE_SIZE];
    __shared__ float densities[BLOCK_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // Load node's edges into shared memory
    int start = row_ptr[tid];
    int end = row_ptr[tid + 1];
    int degree = min(end - start, MAX_QUEUE_SIZE);
    
    for (int i = threadIdx.x; i < degree; i += blockDim.x) {
        if (i < MAX_QUEUE_SIZE) {
            shared_edges[i] = col_idx[start + i];
        }
    }
    __syncthreads();
    
    // Compute local density score
    float density = compute_density(row_ptr, col_idx, tid, start, start + degree);
    densities[threadIdx.x] = density;
    __syncthreads();
    
    // Determine partition based on density and current sizes
    float density_rank = density * (num_partitions - 1);
    int target_partition = min((int)density_rank, num_partitions - 1);
    
    // Try to assign to target partition or find next available
    for (int offset = 0; offset < num_partitions; offset++) {
        int partition = (target_partition + offset) % num_partitions;
        unsigned int old_size = atomicAdd(&partition_sizes[partition], 1);
        
        if (old_size < max_partition_size) {
            partition_labels[tid] = partition;
            if (offset > 0) {
                atomicSub(&partition_sizes[partition], 1);
            }
            break;
        }
        atomicSub(&partition_sizes[partition], 1);
    }
}
"""

def gpu_partition_graph(adjacency_matrix, num_clusters, feature_matrix=None, max_cluster_size=None):
    """Optimized graph partitioning using density-based clustering"""
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
    
    d_row_ptr = cuda.mem_alloc(row_ptr.nbytes)
    d_col_idx = cuda.mem_alloc(col_idx.nbytes)
    d_partition_labels = cuda.mem_alloc(partition_labels.nbytes)
    d_partition_sizes = cuda.mem_alloc(partition_sizes.nbytes)
    
    # Copy data to GPU
    cuda.memcpy_htod(d_row_ptr, row_ptr)
    cuda.memcpy_htod(d_col_idx, col_idx)
    cuda.memcpy_htod(d_partition_labels, partition_labels)
    cuda.memcpy_htod(d_partition_sizes, partition_sizes)
    
    # Configure and launch kernel
    block_size = min(256, num_nodes)
    grid_size = (num_nodes + block_size - 1) // block_size
    
    try:
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
        
        # Create clusters
        clusters = [[] for _ in range(num_clusters)]
        for node, label in enumerate(partition_labels):
            if 0 <= label < num_clusters:
                clusters[label].append(node)
        
        # Validate clusters
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
