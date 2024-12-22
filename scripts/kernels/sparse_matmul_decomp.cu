__device__ inline int binary_search(const int* array, int left, int right, int target) {
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (array[mid] == target) return mid;
        if (array[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

__global__ void sparse_matmul(const float* __restrict__ A_data,
                              const int* __restrict__ A_indices,
                              const int* __restrict__ A_indptr,
                              const float* __restrict__ B_data,
                              const int* __restrict__ B_indices,
                              const int* __restrict__ B_indptr,
                              float* __restrict__ C,
                              const int num_rows,
                              const int num_cols,
                              const int num_cols_B) {
    __shared__ float shared_sum[32][32];

    const int row = blockIdx.y * 32 + threadIdx.y;
    const int col = blockIdx.x * 32 + threadIdx.x;

    if (row < num_rows && col < num_cols_B) {
        float sum = 0.0f;
        const int row_start = A_indptr[row];
        const int row_end = A_indptr[row + 1];

#pragma unroll 4
        for (int idx = row_start; idx < row_end; ++idx) {
            const int k = A_indices[idx];
            const float a_val = A_data[idx];
            const int col_start = B_indptr[k];
            const int col_end = B_indptr[k + 1];

            const int pos = binary_search(B_indices, col_start, col_end - 1, col);
            if (pos != -1) {
                sum += a_val * B_data[pos];
            }
        }

        shared_sum[threadIdx.y][threadIdx.x] = sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            float final_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                final_sum += shared_sum[threadIdx.y][i];
            }
            C[row * num_cols_B + col] = final_sum;
        }
    }
}