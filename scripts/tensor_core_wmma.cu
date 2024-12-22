%%cuda_group_save -g "tensor_core5" -n "tensor_core_wmma5.cu"

#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

using namespace nvcuda;

// Dimensions for WMMA tiles
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Kernel for Tensor Core matrix multiplication
__global__ void tensorCoreMatMul(half *A, half *B, float *C, int M, int N, int K) {
    // Define fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize output fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / 32;

    if (warpM < M / WMMA_M && warpN < N / WMMA_N) {
        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + warpN * WMMA_N, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
    }
}

// Function to load a dense matrix from a text file
void loadDenseMatrixFromTxt(const std::string &filename, std::vector<half> &matrix, int &rows, int &cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the matrix file.");
    }

    file >> rows >> cols; // First two values in the file are the dimensions

    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        float value;
        file >> value;
        matrix[i] = __float2half(value);
    }

    file.close();
}

int main() {
    const std::string matrix_file = "matrix.txt"; // Replace with your matrix file path
    const int dense_features = 1000;             // Number of features in the dense matrix

    // Load sparse matrix from file
    std::vector<half> h_sparse_matrix;
    int sparse_rows, sparse_cols;
    loadDenseMatrixFromTxt(matrix_file, h_sparse_matrix, sparse_rows, sparse_cols);

    // Initialize dense feature matrix
    std::vector<half> h_dense_matrix;
    h_dense_matrix.resize(sparse_cols * dense_features);
    for (auto &val : h_dense_matrix) {
        val = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }

    // Allocate GPU memory
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sparse_rows * sparse_cols * sizeof(half));
    cudaMalloc(&d_B, sparse_cols * dense_features * sizeof(half));
    cudaMalloc(&d_C, sparse_rows * dense_features * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_sparse_matrix.data(), sparse_rows * sparse_cols * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_dense_matrix.data(), sparse_cols * dense_features * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sparse_rows * dense_features * sizeof(float));

    // Configure kernel launch parameters
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((sparse_rows + WMMA_M - 1) / WMMA_M, (dense_features + WMMA_N - 1) / WMMA_N);

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tensorCoreMatMul<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, sparse_rows, dense_features, sparse_cols);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    // Calculate time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Matrix multiplication completed in " << milliseconds << " ms." << std::endl;

    // Copy result back to host
    std::vector<float> h_result(sparse_rows * dense_features);
    cudaMemcpy(h_result.data(), d_C, sparse_rows * dense_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
