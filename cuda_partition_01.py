import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import scipy.sparse as sp

PARTITION_KERNELS = """
// Constants for kernel configuration
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_QUEUE_SIZE 4096

// Shared memory queue structure
typedef struct {
    int data[MAX_QUEUE_SIZE];
    unsigned int head;
    unsigned int tail;
} Queue;

// Atomic queue operations
__device__ bool queue_push(Queue* q, int value) {
    unsigned int tail = atomicAdd(&q->tail, 1);
    if (tail >= MAX_QUEUE_SIZE) {
        atomicSub(&q->tail, 1);
        return false;
    }
    q->data[tail % MAX_QUEUE_SIZE] = value;
    return true;
}

__device__ int queue_pop(Queue* q) {
    unsigned int head = atomicAdd(&q->head, 1);
    if (head >= q->tail) {
        atomicSub(&q->head, 1);
        return -1;
    }
    return q->data[head % MAX_QUEUE_SIZE];
}

// BFS kernel for cluster expansion
__global__ void bfs_cluster_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ cluster_assignments,
    unsigned int* __restrict__ cluster_sizes,
    int* __restrict__ frontier,
    unsigned int* frontier_size,
    const unsigned int max_cluster_size,
    const int num_nodes,
    const int cluster_id
) {
    __shared__ Queue queue;
    if (threadIdx.x == 0) {
        queue.head = 0;
        queue.tail = *frontier_size;
        for (int i = 0; i < *frontier_size; i++) {
            queue.data[i] = frontier[i];
        }
    }
    __syncthreads();
    
    while (true) {
        __syncthreads();
        if (queue.head >= queue.tail) break;
        
        int node = queue_pop(&queue);
        if (node == -1) continue;
        
        // Process neighbors
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            int neighbor = col_idx[edge];
            
            // Try to claim this node for current cluster
            if (cluster_assignments[neighbor] == -1) {
                if (atomicCAS(&cluster_assignments[neighbor], -1, cluster_id) == -1) {
                    unsigned int size = atomicAdd(&cluster_sizes[cluster_id], 1);
                    if (size < max_cluster_size) {
                        queue_push(&queue, neighbor);
                    }
                }
            }
        }
    }
    
    // Update frontier for next iteration if needed
    if (threadIdx.x == 0) {
        *frontier_size = queue.tail - queue.head;
        for (unsigned int i = queue.head; i < queue.tail; i++) {
            frontier[i - queue.head] = queue.data[i % MAX_QUEUE_SIZE];
        }
    }
}

// Initialize clusters based on degree centrality
__global__ void initialize_clusters(
    const int* __restrict__ row_ptr,
    int* __restrict__ cluster_assignments,
    unsigned int* __restrict__ node_degrees,
    const int num_nodes
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        // Calculate node degree
        node_degrees[tid] = row_ptr[tid + 1] - row_ptr[tid];
        cluster_assignments[tid] = -1;
    }
}

// Kernel to find edge cuts using weighted edges
__global__ void find_edge_cuts(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ edge_weights,
    int* __restrict__ cut_markers,
    const int num_nodes,
    const float threshold
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        // Calculate local clustering coefficient
        float local_coeff = 0.0f;
        int degree = end - start;
        
        if (degree > 1) {
            for (int i = start; i < end; i++) {
                int neighbor1 = col_idx[i];
                for (int j = start; j < end; j++) {
                    int neighbor2 = col_idx[j];
                    if (neighbor1 != neighbor2) {
                        // Check if neighbors are connected
                        int n1_start = row_ptr[neighbor1];
                        int n1_end = row_ptr[neighbor1 + 1];
                        for (int k = n1_start; k < n1_end; k++) {
                            if (col_idx[k] == neighbor2) {
                                local_coeff += 1.0f;
                                break;
                            }
                        }
                    }
                }
            }
            
            // Normalize coefficient
            local_coeff /= (float)(degree * (degree - 1));
            
            // Mark edges with low clustering coefficient as potential cuts
            if (local_coeff < threshold) {
                for (int i = start; i < end; i++) {
                    edge_weights[i] = local_coeff;
                    cut_markers[i] = 1;
                }
            }
        }
    }
}

// Modified BFS kernel that grows away from cuts
__global__ void bfs_away_from_cuts(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ cut_markers,
    int* __restrict__ cluster_assignments,
    unsigned int* __restrict__ cluster_sizes,
    int* __restrict__ frontier,
    unsigned int* frontier_size,
    const unsigned int max_cluster_size,
    const int num_nodes,
    const int cluster_id
) {
    __shared__ Queue queue;
    if (threadIdx.x == 0) {
        queue.head = 0;
        queue.tail = *frontier_size;
        for (int i = 0; i < *frontier_size; i++) {
            queue.data[i] = frontier[i];
        }
    }
    __syncthreads();
    
    while (true) {
        __syncthreads();
        if (queue.head >= queue.tail) break;
        
        int node = queue_pop(&queue);
        if (node == -1) continue;
        
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        
        // Process neighbors, preferring those not across cuts
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            int neighbor = col_idx[edge];
            if (cluster_assignments[neighbor] == -1) {
                // Only expand if not marked as cut or if no other options
                if (!cut_markers[edge]) {
                    if (atomicCAS(&cluster_assignments[neighbor], -1, cluster_id) == -1) {
                        unsigned int size = atomicAdd(&cluster_sizes[cluster_id], 1);
                        if (size < max_cluster_size) {
                            queue_push(&queue, neighbor);
                        }
                    }
                }
            }
        }
    }
    
    // Update frontier
    if (threadIdx.x == 0) {
        *frontier_size = queue.tail - queue.head;
        for (unsigned int i = queue.head; i < queue.tail; i++) {
            frontier[i - queue.head] = queue.data[i % MAX_QUEUE_SIZE];
        }
    }
}
"""

def gpu_partition_graph(adjacency_matrix, num_clusters, feature_matrix=None, max_cluster_size=None):
    """
    Partition graph using edge-cut guided BFS clustering and return CSR formatted cluster data
    """
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()
    
    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz
    if max_cluster_size is None:
        max_cluster_size = min(num_nodes // num_clusters * 2, 4096)
    
    # Compile kernels
    mod = SourceModule(PARTITION_KERNELS)
    cut_kernel = mod.get_function("find_edge_cuts")
    bfs_kernel = mod.get_function("bfs_away_from_cuts")
    
    # Allocate GPU memory
    edge_weights = np.zeros(num_edges, dtype=np.float32)
    cut_markers = np.zeros(num_edges, dtype=np.int32)
    cluster_assignments = np.full(num_nodes, -1, dtype=np.int32)
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)
    
    gpu_arrays = {
        'row_ptr': cuda.mem_alloc(adjacency_matrix.indptr.astype(np.int32).nbytes),
        'col_idx': cuda.mem_alloc(adjacency_matrix.indices.astype(np.int32).nbytes),
        'edge_weights': cuda.mem_alloc(edge_weights.nbytes),
        'cut_markers': cuda.mem_alloc(cut_markers.nbytes),
        'cluster_assignments': cuda.mem_alloc(cluster_assignments.nbytes),
        'cluster_sizes': cuda.mem_alloc(cluster_sizes.nbytes)
    }
    
    # Copy data to GPU
    for name, arr in [
        ('row_ptr', adjacency_matrix.indptr.astype(np.int32)),
        ('col_idx', adjacency_matrix.indices.astype(np.int32))
    ]:
        cuda.memcpy_htod(gpu_arrays[name], arr)
    
    # Find edge cuts
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size
    cut_threshold = 0.1  # Adjust this threshold based on your needs
    
    cut_kernel(
        gpu_arrays['row_ptr'],
        gpu_arrays['col_idx'],
        gpu_arrays['edge_weights'],
        gpu_arrays['cut_markers'],
        np.int32(num_nodes),
        np.float32(cut_threshold),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    
    # Get cut information back
    cuda.memcpy_dtoh(cut_markers, gpu_arrays['cut_markers'])
    
    # Find seed nodes near cuts
    cut_edges = np.where(cut_markers == 1)[0]
    seed_nodes = []
    for edge in cut_edges:
        # Find nodes on both sides of cut
        row_idx = np.searchsorted(adjacency_matrix.indptr[:-1], edge, side='right') - 1
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
        cuda.memcpy_htod(gpu_arrays['cluster_assignments'], cluster_assignments)
        
        # Run BFS
        while frontier_size[0] > 0:
            bfs_kernel(
                gpu_arrays['row_ptr'],
                gpu_arrays['col_idx'],
                gpu_arrays['cut_markers'],
                gpu_arrays['cluster_assignments'],
                gpu_arrays['cluster_sizes'],
                frontier_gpu,
                frontier_size_gpu,
                np.uint32(max_cluster_size),
                np.int32(num_nodes),
                np.int32(cluster_id),
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            cuda.memcpy_dtoh(frontier_size, frontier_size_gpu)
    
    # Get final assignments
    cuda.memcpy_dtoh(cluster_assignments, gpu_arrays['cluster_assignments'])
    
    # Handle unassigned nodes
    unassigned = np.where(cluster_assignments == -1)[0]
    for node in unassigned:
        # Assign to nearest cluster
        neighbors = adjacency_matrix[node].indices
        assigned_neighbors = [n for n in neighbors if cluster_assignments[n] >= 0]
        if assigned_neighbors:
            cluster_assignments[node] = cluster_assignments[assigned_neighbors[0]]
        else:
            cluster_assignments[node] = 0
    
    # Create initial clusters from assignments
    clusters = [[] for _ in range(num_clusters)]
    for node, cluster in enumerate(cluster_assignments):
        if cluster >= 0 and cluster < num_clusters:
            clusters[cluster].append(node)
    
    # Create final clusters with features
    cluster_data = []
    for cluster_id, nodes in enumerate(clusters):
        if len(nodes) < 2:  # Skip tiny clusters
            continue
            
        # Extract submatrix for this cluster
        sub_adj = adjacency_matrix[nodes, :][:, nodes]
        
        # Get features for these nodes
        if feature_matrix is not None:
            sub_feat = feature_matrix[nodes, :]
        else:
            sub_feat = adjacency_matrix[nodes, :]  # Fallback to using adjacency as features
        
        # Convert to CSR format if needed
        if not sp.isspmatrix_csr(sub_adj):
            sub_adj = sub_adj.tocsr()
        if not sp.isspmatrix_csr(sub_feat):
            sub_feat = sub_feat.tocsr()
            
        # Store cluster data tuple
        cluster_data.append((sub_adj, sub_feat, nodes, range(adjacency_matrix.shape[0])))
    
    # Clean up
    for arr in gpu_arrays.values():
        arr.free()
    frontier_gpu.free()
    frontier_size_gpu.free()
    
    return cluster_data
