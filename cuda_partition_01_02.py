import numpy as np
import scipy.sparse as sp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from scipy.sparse.linalg import eigsh

QUEUE_SIZE = 16384
MAX_BATCHES = 32

PARTITION_KERNELS = """
// Constants and shared memory optimizations
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define QUEUE_SIZE 16384  // Increased from 8192 to 16384
#define MAX_BATCHES 32    // Increased from 16 to 32

// Enhanced queue with batched processing
typedef struct {
    unsigned int data[QUEUE_SIZE];  // Changed from int to unsigned int
    unsigned int sizes[MAX_BATCHES];
    unsigned int batch_head;
    unsigned int batch_tail;
    unsigned int current_size;
} BatchQueue;

__device__ void init_batch_queue(BatchQueue* q) {
    q->batch_head = 0;
    q->batch_tail = 0;
    q->current_size = 0;
    for (int i = 0; i < MAX_BATCHES; i++) {
        q->sizes[i] = 0;
    }
}

__device__ bool queue_push_batch(BatchQueue* q, unsigned int value) {  // Changed parameter type
    if (q->current_size >= QUEUE_SIZE) return false;
    unsigned int idx = atomicAdd(&q->sizes[q->batch_tail], 1);
    if (idx >= QUEUE_SIZE / MAX_BATCHES) {
        atomicSub(&q->sizes[q->batch_tail], 1);
        return false;
    }
    q->data[q->batch_tail * (QUEUE_SIZE / MAX_BATCHES) + idx] = value;
    atomicAdd(&q->current_size, 1);
    return true;
}

__device__ int queue_pop_batch(BatchQueue* q) {
    if (q->current_size == 0) return -1;
    unsigned int batch_size = q->sizes[q->batch_head];
    if (batch_size == 0) {
        q->batch_head = (q->batch_head + 1) % MAX_BATCHES;
        return -1;
    }
    unsigned int idx = atomicSub(&q->sizes[q->batch_head], 1);
    if (idx <= 0) return -1;
    atomicSub(&q->current_size, 1);
    return q->data[q->batch_head * (QUEUE_SIZE / MAX_BATCHES) + idx - 1];
}

// Spectral clustering kernel for initial partitioning
__global__ void spectral_clustering_kernel(
    const float* __restrict__ fiedler_vector,
    int* __restrict__ initial_partition,
    const int num_nodes,
    const float split_value
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        initial_partition[tid] = fiedler_vector[tid] > split_value ? 1 : 0;
    }
}

// Enhanced BFS kernel with load balancing
__global__ void balanced_bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ edge_weights,
    int* __restrict__ cluster_assignments,
    unsigned int* __restrict__ cluster_sizes,
    const int max_cluster_size,
    const int num_nodes,
    const int cluster_id,
    BatchQueue* queue
) {
    __shared__ BatchQueue shared_queue;
    if (threadIdx.x == 0) {
        init_batch_queue(&shared_queue);
    }
    __syncthreads();
    
    while (true) {
        __syncthreads();
        if (shared_queue.current_size == 0) break;
        
        int node = queue_pop_batch(&shared_queue);
        if (node == -1) continue;
        
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        float local_density = 0.0f;
        
        // Process neighbors in warps for coalesced memory access
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            int neighbor = col_idx[edge];
            if (cluster_assignments[neighbor] == -1) {
                float weight = edge_weights[edge];
                
                // Atomic compare-and-swap with weight consideration
                if (weight > 0.5f && atomicCAS(&cluster_assignments[neighbor], -1, cluster_id) == -1) {
                    unsigned int size = atomicAdd(&cluster_sizes[cluster_id], 1);
                    if (size < max_cluster_size) {
                        queue_push_batch(&shared_queue, neighbor);
                    }
                }
            }
        }
    }
}

// Edge weight computation with memory coalescing
__global__ void compute_edge_weights_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ edge_weights,
    const int num_nodes
) {
    __shared__ int shared_neighbors[BLOCK_SIZE];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < num_nodes) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        for (int i = start; i < end; i++) {
            int j = col_idx[i];
            int j_start = row_ptr[j];
            int j_end = row_ptr[j + 1];
            
            // Compute structural similarity
            float similarity = 0.0f;
            int intersection = 0;
            int i_idx = start;
            int j_idx = j_start;
            
            while (i_idx < end && j_idx < j_end) {
                if (col_idx[i_idx] < col_idx[j_idx]) i_idx++;
                else if (col_idx[i_idx] > col_idx[j_idx]) j_idx++;
                else {
                    intersection++;
                    i_idx++;
                    j_idx++;
                }
            }
            
            int union_size = (end - start) + (j_end - j_start) - intersection;
            edge_weights[i] = union_size > 0 ? (float)intersection / union_size : 0.0f;
        }
    }
}
"""

def compute_fiedler_vector(adjacency_matrix):
    """Compute Fiedler vector for initial spectral partitioning"""
    laplacian = sp.csgraph.laplacian(adjacency_matrix, normed=True)
    try:
        _, vectors = eigsh(laplacian, k=2, which='SM')
        return vectors[:, 1]  # Fiedler vector
    except:
        print("Failed to compute Fiedler vector")
        return None

def gpu_partition_graph(adjacency_matrix, num_clusters, feature_matrix=None, max_cluster_size=None):
    """Optimized graph partitioning for sparse matrix multiplication"""
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()
    
    num_nodes = adjacency_matrix.shape[0]
    if max_cluster_size is None:
        if feature_matrix is not None:
            feature_dim = feature_matrix.shape[1]
            max_cluster_size = min(
                int(np.sqrt(num_nodes * feature_dim)), 
                num_nodes // num_clusters * 2
            )
        else:
            max_cluster_size = num_nodes // num_clusters * 2
    
    # Initialize CUDA kernels
    mod = SourceModule(PARTITION_KERNELS)
    compute_weights = mod.get_function("compute_edge_weights_kernel")
    spectral_kernel = mod.get_function("spectral_clustering_kernel")
    bfs_kernel = mod.get_function("balanced_bfs_kernel")
    
    # Compute initial partitioning using spectral clustering
    fiedler = compute_fiedler_vector(adjacency_matrix)
    if fiedler is None:
        raise RuntimeError("Failed to compute Fiedler vector")
    
    # Prepare GPU arrays
    d_row_ptr = cuda.mem_alloc(adjacency_matrix.indptr.astype(np.int32).nbytes)
    d_col_idx = cuda.mem_alloc(adjacency_matrix.indices.astype(np.int32).nbytes)
    d_fiedler = cuda.mem_alloc(fiedler.astype(np.float32).nbytes)
    d_assignments = cuda.mem_alloc(num_nodes * 4)
    d_sizes = cuda.mem_alloc(num_clusters * 4)
    d_edge_weights = cuda.mem_alloc(adjacency_matrix.nnz * 4)
    
    # Allocate and initialize queue
    queue_struct_size = 4 * (QUEUE_SIZE + MAX_BATCHES + 3)  # data + sizes + head/tail/current_size
    d_queue = cuda.mem_alloc(queue_struct_size)
    # Initialize queue memory to zeros
    cuda.memset_d32(d_queue, 0, queue_struct_size // 4)
    
    # Copy data to GPU
    cuda.memcpy_htod(d_row_ptr, adjacency_matrix.indptr.astype(np.int32))
    cuda.memcpy_htod(d_col_idx, adjacency_matrix.indices.astype(np.int32))
    cuda.memcpy_htod(d_fiedler, fiedler.astype(np.float32))
    
    # Initialize assignments and sizes to -1 and 0 respectively
    assignments = np.full(num_nodes, -1, dtype=np.int32)
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)
    cuda.memcpy_htod(d_assignments, assignments)
    cuda.memcpy_htod(d_sizes, cluster_sizes)
    
    # Compute edge weights
    compute_weights(
        d_row_ptr, d_col_idx, d_edge_weights, np.int32(num_nodes),
        block=(256, 1, 1), grid=((num_nodes + 255) // 256, 1)
    )
    
    # Process clusters with BFS
    for cluster_id in range(num_clusters):
        # Reset queue for each cluster
        cuda.memset_d32(d_queue, 0, queue_struct_size // 4)
        
        # Launch BFS kernel with proper arguments
        bfs_kernel(
            d_row_ptr, d_col_idx, d_edge_weights,
            d_assignments, d_sizes,
            np.int32(max_cluster_size),
            np.int32(num_nodes),
            np.int32(cluster_id),
            d_queue,
            block=(256, 1, 1),
            grid=((num_nodes + 255) // 256, 1)
        )
    
    # Get results
    cuda.memcpy_dtoh(assignments, d_assignments)
    
    # Clean up
    for arr in [d_row_ptr, d_col_idx, d_fiedler, d_assignments, d_sizes, d_edge_weights, d_queue]:
        arr.free()
    
    # Validate and create clusters
    clusters = [[] for _ in range(num_clusters)]
    for node, cluster in enumerate(assignments):
        if cluster >= 0 and cluster < num_clusters:
            clusters[cluster].append(node)
        else:
            raise RuntimeError(f"Invalid cluster assignment {cluster} for node {node}")
    
    # Validate cluster sizes
    if any(len(c) == 0 for c in clusters):
        raise RuntimeError("Some clusters are empty")
        
    return clusters
