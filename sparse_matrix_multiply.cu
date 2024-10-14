#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <chrono>
#include <vector>

namespace py = pybind11;

__global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, float *B_data, int *B_indices, int *B_indptr, float *C, int num_rows, int num_cols, int num_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows && col < num_cols_B) {
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
        C[row * num_cols_B + col] = sum;
    }
}

std::tuple<py::array_t<float>, double, double> sparse_matrix_multiply(
    py::array_t<float> A_data, py::array_t<int> A_indices, py::array_t<int> A_indptr,
    py::array_t<float> B_data, py::array_t<int> B_indices, py::array_t<int> B_indptr,
    int num_rows, int num_cols, int num_cols_B) {

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate GPU memory
    float *A_data_gpu, *B_data_gpu, *C_gpu;
    int *A_indices_gpu, *A_indptr_gpu, *B_indices_gpu, *B_indptr_gpu;
    cudaMalloc(&A_data_gpu, A_data.size() * sizeof(float));
    cudaMalloc(&A_indices_gpu, A_indices.size() * sizeof(int));
    cudaMalloc(&A_indptr_gpu, A_indptr.size() * sizeof(int));
    cudaMalloc(&B_data_gpu, B_data.size() * sizeof(float));
    cudaMalloc(&B_indices_gpu, B_indices.size() * sizeof(int));
    cudaMalloc(&B_indptr_gpu, B_indptr.size() * sizeof(int));
    cudaMalloc(&C_gpu, num_rows * num_cols_B * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(A_data_gpu, A_data.data(), A_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A_indices_gpu, A_indices.data(), A_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_indptr_gpu, A_indptr.data(), A_indptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_data_gpu, B_data.data(), B_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_indices_gpu, B_indices.data(), B_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_indptr_gpu, B_indptr.data(), B_indptr.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block_size(16, 16, 1);
    dim3 grid_size((num_cols_B + 15) / 16, (num_rows + 15) / 16, 1);
    sparse_matmul<<<grid_size, block_size>>>(A_data_gpu, A_indices_gpu, A_indptr_gpu, B_data_gpu, B_indices_gpu, B_indptr_gpu, C_gpu, num_rows, num_cols, num_cols_B);

    // Copy the result back to host
    std::vector<float> C_host(num_rows * num_cols_B);
    cudaMemcpy(C_host.data(), C_gpu, num_rows * num_cols_B * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A_data_gpu);
    cudaFree(A_indices_gpu);
    cudaFree(A_indptr_gpu);
    cudaFree(B_data_gpu);
    cudaFree(B_indices_gpu);
    cudaFree(B_indptr_gpu);
    cudaFree(C_gpu);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Return the result and the elapsed time
    return std::make_tuple(py::array_t<float>(C_host.size(), C_host.data()), elapsed.count(), 0.0); // 0.0 for peak memory usage placeholder
}

PYBIND11_MODULE(sparse_matrix_multiply, m) {
    m.def("sparse_matrix_multiply", &sparse_matrix_multiply, "Sparse matrix multiplication using CUDA");
}