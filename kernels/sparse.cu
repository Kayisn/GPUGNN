extern "C" __global__ void matmul(
    const float *A_data, const int *A_indices, const int *A_indptr,
    const float *B_data, const int *B_indices, const int *B_indptr,
    float *C, int num_rows, int num_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        float sum = 0;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int k = A_indices[idx];
            int col_start = B_indptr[k];
            int col_end = B_indptr[k + 1];

            for (int jdx = col_start; jdx < col_end; ++jdx) {
                if (B_indices[jdx] == col) {
                    sum += A_data[idx] * B_data[jdx];
                    break;
                }
            }
        }
        C[row * num_cols + col] = sum;
    }
}