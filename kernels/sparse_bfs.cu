extern "C" __global__ void matmul(float *A_data, int *A_indices, int *A_indptr, 
                            float *B_data, int *B_indices, int *B_indptr, 
                            float *C, int num_rows, int num_cols, int num_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols_B) {
        float sum = 0.0f;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];
        
        // Coalesced memory access pattern for Maxwell
        for (int idx = row_start; idx < row_end; ++idx) {
            int k = A_indices[idx];
            float a_val = A_data[idx];
            
            int col_start = B_indptr[k];
            int col_end = B_indptr[k + 1];
            
            // Binary search for matching column
            int left = col_start;
            int right = col_end - 1;
            
            while (left <= right) {
                int mid = (left + right) >> 1;
                int bcol = B_indices[mid];
                if (bcol == col) {
                    sum += a_val * B_data[mid];
                    break;
                }
                if (bcol < col) left = mid + 1;
                else right = mid - 1;
            }
        }
        C[row * num_cols_B + col] = sum;
    }
}