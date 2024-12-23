#define TILE_SIZE 32
#define BLOCK_SIZE 32

extern "C" __global__ void matmul(
    const float *A_data, const int *A_indices, const int *A_indptr,  // A in CSR
    const float *B_data, const int *B_indices, const int *B_indptr,  // B in CSC
    float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
    __shared__ float A_tile[TILE_SIZE][BLOCK_SIZE];  // Tile for A values
    __shared__ int A_cols[TILE_SIZE][BLOCK_SIZE];    // Tile for A column indices
    __shared__ float B_tile[TILE_SIZE][BLOCK_SIZE];  // Tile for B values
    __shared__ int B_rows[TILE_SIZE][BLOCK_SIZE];    // Tile for B row indices

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Calculate number of tiles needed
    int num_tiles = (num_cols_A + TILE_SIZE - 1) / TILE_SIZE;

    // Process one tile at a time
    for (int t = 0; t < num_tiles; t++) {
        // Clear shared memory tiles
        A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        A_cols[threadIdx.y][threadIdx.x] = -1;
        B_rows[threadIdx.y][threadIdx.x] = -1;
        __syncthreads();

        // Load A tile - each thread loads one element
        if (row < num_rows_A) {
            int row_start = A_indptr[row];
            int row_end = A_indptr[row + 1];

            for (int i = row_start; i < row_end; i++) {
                int col_idx = A_indices[i];
                // Check if element belongs to current tile
                if (col_idx >= t * TILE_SIZE && col_idx < (t + 1) * TILE_SIZE) {
                    int tile_idx = col_idx - t * TILE_SIZE;
                    A_tile[threadIdx.y][tile_idx] = A_data[i];
                    A_cols[threadIdx.y][tile_idx] = col_idx;
                }
            }
        }

        // Load B tile - each thread loads one element
        if (col < num_cols_B) {
            int col_start = B_indptr[col];
            int col_end = B_indptr[col + 1];

            for (int i = col_start; i < col_end; i++) {
                int row_idx = B_indices[i];
                // Check if element belongs to current tile
                if (row_idx >= t * TILE_SIZE && row_idx < (t + 1) * TILE_SIZE) {
                    int tile_idx = row_idx - t * TILE_SIZE;
                    B_tile[tile_idx][threadIdx.x] = B_data[i];
                    B_rows[tile_idx][threadIdx.x] = row_idx;
                }
            }
        }
        __syncthreads();

        // Compute partial products for this tile
        if (row < num_rows_A && col < num_cols_B) {
            for (int k = 0; k < TILE_SIZE; k++) {
                if (A_cols[threadIdx.y][k] == B_rows[k][threadIdx.x] &&
                    A_cols[threadIdx.y][k] != -1) {
                    sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    // Store final result
    if (row < num_rows_A && col < num_cols_B) {
        C[row * num_cols_B + col] = sum;
    }
}