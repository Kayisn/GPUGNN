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

def estimate_gpu_capabilities():
    """Estimate GPU capabilities for optimizing partition sizes"""
    device = cuda.Device(0)
    attributes = device.get_attributes()
    
    return {
        'max_threads_block': attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK],
        'max_grid_dims': [
            attributes[cuda.device_attribute.MAX_GRID_DIM_X],
            attributes[cuda.device_attribute.MAX_GRID_DIM_Y],
            attributes[cuda.device_attribute.MAX_GRID_DIM_Z]
        ],
        'max_shared_memory': attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK],
        'num_sms': attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT],
        'compute_capability': device.compute_capability(),
        'total_memory': device.total_memory(),
        'warp_size': attributes[cuda.device_attribute.WARP_SIZE]
    }

def calculate_optimal_cluster_size(graph_properties, gpu_caps):
    """Calculate optimal cluster size based on graph and GPU properties"""
    avg_degree = graph_properties['avg_degree']
    density = graph_properties['density']
    max_degree = graph_properties['max_degree']
    
    # Calculate memory constraints
    mem_per_edge = 4 * 3  # 12 bytes per edge (2 integers + 1 float)
    mem_per_node = 4  # 4 bytes per node (1 integer)
    available_mem = gpu_caps['total_memory'] * 0.8  # Use 80% of GPU memory
    
    # Calculate compute constraints
    threads_per_block = min(256, gpu_caps['max_threads_block'])
    max_concurrent_threads = gpu_caps['num_sms'] * threads_per_block * 32
    
    # Estimate optimal cluster size based on memory and compute constraints
    max_edges_memory = available_mem / mem_per_edge
    max_edges_compute = max_concurrent_threads * avg_degree
    
    # Consider graph density in calculations
    density_factor = np.sqrt(density)
    optimal_size = min(
        int(max_edges_memory * density_factor),
        int(max_edges_compute * density_factor),
        int(max_concurrent_threads / avg_degree)
    )
    
    return max(100, min(optimal_size, 4096))  # Ensure reasonable bounds

def analyze_graph_properties(adjacency_matrix, sample_size=1000):
    """Analyze graph properties for optimizing partitioning"""
    num_nodes = adjacency_matrix.shape[0]
    num_edges = adjacency_matrix.nnz
    
    # Calculate basic properties
    degrees = np.diff(adjacency_matrix.indptr)
    avg_degree = num_edges / num_nodes
    density = num_edges / (num_nodes * num_nodes)
    
    # Sample nodes for detailed analysis
    sample_indices = np.random.choice(num_nodes, min(sample_size, num_nodes), replace=False)
    sampled_degrees = degrees[sample_indices]
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'density': density,
        'max_degree': np.max(degrees),
        'degree_variance': np.var(sampled_degrees),
        'degree_distribution': np.percentile(sampled_degrees, [25, 50, 75, 90])
    }

def gpu_partition_graph(adjacency_matrix, kernel_manager=None, feature_matrix=None):
    """Enhanced graph partitioning using GPU-accelerated BFS with dynamic sizing"""
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()
    
    # Initialize kernel manager if not provided
    if kernel_manager is None:
        print("Need to include the kernel manager")
        raise ValueError("Kernel manager is required for GPU partitioning")
    
    # Analyze graph and GPU properties
    graph_props = analyze_graph_properties(adjacency_matrix)
    gpu_caps = estimate_gpu_capabilities()
    
    # Calculate optimal cluster parameters
    optimal_cluster_size = calculate_optimal_cluster_size(graph_props, gpu_caps)
    num_clusters = max(2, int(graph_props['num_nodes'] / optimal_cluster_size))
    
    print(f"\nPartitioning Configuration:")
    print(f"Optimal cluster size: {optimal_cluster_size}")
    print(f"Number of clusters: {num_clusters}")
    
    # Initialize arrays
    cluster_assignments = np.full(graph_props['num_nodes'], -1, dtype=np.int32)
    cluster_sizes = np.zeros(num_clusters, dtype=np.uint32)
    
    # Allocate GPU memory
    try:
        gpu_arrays = {
            'row_ptr': cuda.mem_alloc(adjacency_matrix.indptr.astype(np.int32).nbytes),
            'col_idx': cuda.mem_alloc(adjacency_matrix.indices.astype(np.int32).nbytes),
            'assignments': cuda.mem_alloc(cluster_assignments.nbytes),
            'sizes': cuda.mem_alloc(cluster_sizes.nbytes)
        }
        
        # Copy data to GPU
        for name, arr in [
            ('row_ptr', adjacency_matrix.indptr.astype(np.int32)),
            ('col_idx', adjacency_matrix.indices.astype(np.int32))
        ]:
            cuda.memcpy_htod(gpu_arrays[name], arr)
        
        # Set up kernel parameters
        block_size = min(256, gpu_caps['max_threads_block'])
        grid_size = (graph_props['num_nodes'] + block_size - 1) // block_size
        
        # Execute partitioning
        bfs_kernel = kernel_manager.get_kernel('balanced_bfs')
        if not bfs_kernel:
            raise RuntimeError("Failed to get BFS kernel")
        
        # Process clusters
        clusters = []
        for cluster_id in range(num_clusters):
            # Find seed node
            seed = np.random.randint(0, graph_props['num_nodes'])
            while cluster_assignments[seed] != -1:
                seed = np.random.randint(0, graph_props['num_nodes'])
            
            cluster_assignments[seed] = cluster_id
            
            # Launch BFS kernel
            bfs_kernel(
                gpu_arrays['row_ptr'],
                gpu_arrays['col_idx'],
                gpu_arrays['assignments'],
                gpu_arrays['sizes'],
                np.int32(optimal_cluster_size),
                np.int32(graph_props['num_nodes']),
                np.int32(cluster_id),
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            # Update assignments
            cuda.memcpy_dtoh(cluster_assignments, gpu_arrays['assignments'])
            
            # Collect cluster nodes
            cluster_nodes = np.where(cluster_assignments == cluster_id)[0]
            if len(cluster_nodes) > 0:
                clusters.append(cluster_nodes.tolist())
        
        return clusters
        
    except cuda.Error as e:
        print(f"CUDA error: {e}")
        raise
    finally:
        # Cleanup GPU memory
        for arr in gpu_arrays.values():
            try:
                arr.free()
            except cuda.Error:
                pass
