__global__ void pagerank_iteration(
    const float *in_rank,
    float *out_rank,
    const int *row_ptr,
    const int *col_idx,
    const float *values,
    const float damping,
    const int num_nodes) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;

    if (node < num_nodes) {
        float sum = 0.0f;
        int start = row_ptr[node];
        int end = row_ptr[node + 1];

        // Sum contributions from incoming edges
        for (int edge = start; edge < end; edge++) {
            int src = col_idx[edge];
            int src_degree = row_ptr[src + 1] - row_ptr[src];
            if (src_degree > 0) {
                sum += in_rank[src] / src_degree;
            }
        }

        // Apply damping factor
        out_rank[node] = (1.0f - damping) / num_nodes + damping * sum;
    }
}