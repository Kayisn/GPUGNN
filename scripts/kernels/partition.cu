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
    const int cluster_id) {
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
    const int num_nodes) {
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
    const float threshold) {
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
    const int cluster_id) {
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