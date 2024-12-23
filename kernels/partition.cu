// Ensure extern "C" for proper name mangling
extern "C" {

// Constants and shared memory optimizations
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define QUEUE_SIZE 1024  // Reduced queue size
#define MAX_BATCHES 16   // Reduced batch count
#define SHARED_MEM_ALIGN 16

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
__global__ void spectral_clustering(
    const float* __restrict__ fiedler_vector,
    int* __restrict__ initial_partition,
    const int num_nodes,
    const float split_value) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        initial_partition[tid] = fiedler_vector[tid] > split_value ? 1 : 0;
    }
}

// Enhanced BFS kernel with load balancing
__global__ void balanced_bfs(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ edge_weights,
    int* __restrict__ cluster_assignments,
    unsigned int* __restrict__ cluster_sizes,
    const int max_edges_per_cluster,
    const int num_nodes,
    const int cluster_id,
    BatchQueue* queue) {
    __shared__ BatchQueue shared_queue;
    __shared__ unsigned int cluster_edge_count;

    if (threadIdx.x == 0) {
        init_batch_queue(&shared_queue);
        cluster_edge_count = 0;
    }
    __syncthreads();

    while (true) {
        __syncthreads();
        if (shared_queue.current_size == 0 || cluster_edge_count >= max_edges_per_cluster) break;

        int node = queue_pop_batch(&shared_queue);
        if (node == -1) continue;

        int start = row_ptr[node];
        int end = row_ptr[node + 1];

        // Count edges before processing
        int edge_count = atomicAdd(&cluster_edge_count, end - start);
        if (edge_count >= max_edges_per_cluster) continue;

        // Process neighbors with edge density consideration
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            int neighbor = col_idx[edge];
            if (cluster_assignments[neighbor] == -1) {
                float weight = edge_weights[edge];

                // Use edge weight as density metric
                if (weight > 0.5f && atomicCAS(&cluster_assignments[neighbor], -1, cluster_id) == -1) {
                    atomicAdd(&cluster_sizes[cluster_id], 1);
                    queue_push_batch(&shared_queue, neighbor);
                }
            }
        }
    }
}

// Edge weight computation with memory coalescing
__global__ void compute_edge_weights(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ edge_weights,
    const int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];

        for (int i = start; i < end; i++) {
            int j = col_idx[i];
            int j_start = row_ptr[j];
            int j_end = row_ptr[j + 1];

            // Compute edge density metric
            int local_edges = end - start;
            int neighbor_edges = j_end - j_start;
            float density = (float)(local_edges + neighbor_edges) /
                            (float)(num_nodes * 2);  // Normalize by max possible edges

            edge_weights[i] = density;
        }
    }
}

}  // extern "C"