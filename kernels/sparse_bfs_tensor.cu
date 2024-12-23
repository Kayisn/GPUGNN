#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void matmul(
    __half *A_data, int *A_indices, int *A_indptr,
    __half *B_data, int *B_indices, int *B_indptr,
    float *C, int num_rows, int num_cols, int num_cols_B) {
    // Define tile sizes for tensor cores
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Shared memory for the tiles
    __shared__ __half a_tile[WMMA_M][WMMA_K];
    __shared__ __half b_tile[WMMA_K][WMMA_N];

    // Initialize accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols_B) {
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];

        // Load and multiply using tensor cores where possible
        for (int idx = row_start; idx < row_end; ++idx) {
            int k = A_indices[idx];
            __half a_val = A_data[idx];

            int col_start = B_indptr[k];
            int col_end = B_indptr[k + 1];

            // Binary search for matching column
            int left = col_start;
            int right = col_end - 1;

            while (left <= right) {
                int mid = (left + right) >> 1;
                int bcol = B_indices[mid];
                if (bcol == col) {
                    // Load tiles into shared memory
                    if (threadIdx.x < WMMA_K) {
                        a_tile[threadIdx.y][threadIdx.x] = a_val;
                        b_tile[threadIdx.x][threadIdx.y] = B_data[mid];
                    }

                    __syncthreads();

                    // Create matrix fragments with explicit layout
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

                    // Load fragments
                    wmma::load_matrix_sync(a_frag, &a_tile[0][0], WMMA_K);
                    wmma::load_matrix_sync(b_frag, &b_tile[0][0], WMMA_N);

                    // Perform matrix multiplication
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

                    __syncthreads();
                    break;
                }
                if (bcol < col)
                    left = mid + 1;
                else
                    right = mid - 1;
            }
        }

        // Store result
        C[row * num_cols_B + col] = acc_frag.x[0];  // Extract first element of accumulator fragment
    }
}
