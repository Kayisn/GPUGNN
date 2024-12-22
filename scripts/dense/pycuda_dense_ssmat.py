import json
import time
import scipy.io
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pathlib import Path

# Print GPU information
def print_gpu_info():
    device = cuda.Device(0)
    print("GPU Information:")                        
    print(f"  Name: {device.name()}")
    print(f"  Compute Capability: {device.compute_capability()}")
    print(f"  Total Memory: {device.total_memory() / 1e9:.2f} GB")

# Matrix multiplication method for dense matrices (one thread per output cell), on  SuiteSparse Matrix Collection. 
def dense_matrix_multiply_pycuda(A, B):
    A_dense = A.astype(np.float32)
    B_dense = B.astype(np.float32)

    rows_A, cols_A = A_dense.shape
    rows_B, cols_B = B_dense.shape

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions do not match for multiplication")

    A_gpu = cuda.mem_alloc(A_dense.nbytes)
    B_gpu = cuda.mem_alloc(B_dense.nbytes)
    C_gpu = cuda.mem_alloc(rows_A * cols_B * A_dense.dtype.itemsize)

    cuda.memcpy_htod(A_gpu, A_dense)
    cuda.memcpy_htod(B_gpu, B_dense)

    mod = SourceModule(
        """
        __global__ void matmul(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < rowsA && col < colsB) {
                float sum = 0.0f;
                for (int k = 0; k < colsA; ++k) {
                    sum += A[row * colsA + k] * B[k * colsB + col];
                }
                C[row * colsB + col] = sum;
            }
        }
        """
    )

    matmul = mod.get_function("matmul")
    block_size = (16, 16, 1)
    grid_size = (
        int(np.ceil(cols_B / 16)),
        int(np.ceil(rows_A / 16)),
        1,
    )
    matmul(
        A_gpu, B_gpu, C_gpu,
        np.int32(rows_A), np.int32(cols_A), np.int32(cols_B),
        block=block_size, grid=grid_size
    )

    C_dense = np.empty((rows_A, cols_B), dtype=np.float32)
    cuda.memcpy_dtoh(C_dense, C_gpu)

    return C_dense

# Perform GPU warmup for fair comparison
def gpu_warmup():
    dummy_A = np.ones((16, 16), dtype=np.float32)
    dummy_B = np.ones((16, 16), dtype=np.float32)
    print("Warming up GPU...")
    for _ in range(2):  # Run the warmup twice
        dense_matrix_multiply_pycuda(dummy_A, dummy_B)
    print("GPU warmup complete.")

# Process all .mtx files in the directory
def process_matrices(matrices_dir):
    results = []
    for mtx_file in Path(matrices_dir).glob("*.mtx"):
        print(f"Processing matrix: {mtx_file}")

        # Load the matrix
        matrix = scipy.io.mmread(mtx_file).astype(np.float32)

        if matrix.shape[0] != matrix.shape[1]:
            print(f"Skipping non-square matrix: {mtx_file}")
            continue

        # Perform A × A multiplication
        try:
            start_time = time.time()
            result = dense_matrix_multiply_pycuda(matrix, matrix)
            elapsed_time = time.time() - start_time

            results.append({
                "matrix_name": mtx_file.name,
                "num_rows": matrix.shape[0],
                "num_cols": matrix.shape[1],
                "time_seconds": elapsed_time,
            })

            print(f"Processed {mtx_file.name} in {elapsed_time:.4f} seconds")
        except Exception as e:
            print(f"Error processing {mtx_file.name}: {e}")

    return results

# Save results to a JSON file
def save_results(results, result_file):
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results have been saved to '{result_file}'")

if __name__ == "__main__":
    print_gpu_info()  # Print GPU details

    matrices_dir = "../ssmatrices"  # Path to the directory containing .mtx files
    result_file = "../results/matrix_multiplication_results.json"

    # GPU warmup
    gpu_warmup()

    # Process matrices
    results = process_matrices(matrices_dir)

    # Save results
    save_results(results, result_file)
