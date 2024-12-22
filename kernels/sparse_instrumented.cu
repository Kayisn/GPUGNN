__global__ void matmul(
    const float *A_data, const int *A_indices, const int *A_indptr,  // A in CSR
    const float *B_data, const int *B_indices, const int *B_indptr,  // B in CSC
    float *C, int *work_counters,                                    // Track operations per thread
    int num_rows_A, int num_cols_A, int num_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = row * num_cols_B + col;
    int ops_count = 0;

    if (row < num_rows_A && col < num_cols_B) {
        float sum = 0.0f;

        // For CSR format of A: A_indptr[row] to A_indptr[row+1] gives elements in this row
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];

        // For CSC format of B: B_indptr[col] to B_indptr[col+1] gives elements in this column
        int col_start = B_indptr[col];
        int col_end = B_indptr[col + 1];

        // Pointers for walking through both sparse representations
        int a_idx = row_start;
        int b_idx = col_start;

        // Merge-join style intersection of row and column
        while (a_idx < row_end && b_idx < col_end) {
            ops_count++;
            int a_col = A_indices[a_idx];  // Column index in A
            int b_row = B_indices[b_idx];  // Row index in B

            if (a_col == b_row) {
                // Matching indices - multiply and add
                sum += A_data[a_idx] * B_data[b_idx];
                a_idx++;
                b_idx++;
            } else if (a_col < b_row) {
                // Need to move forward in A
                a_idx++;
            } else {
                // Need to move forward in B
                b_idx++;
            }
        }

        // Store result - use row-major ordering since output is dense
        C[tid] = sum;
        work_counters[tid] = ops_count;
    }
}