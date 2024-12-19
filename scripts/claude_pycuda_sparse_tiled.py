"""
Key features of this tiled implementation:

Each block processes a TILE_SIZE x TILE_SIZE tile of the output matrix
Uses shared memory to store tiles of both input matrices
Synchronizes threads after loading tiles and after processing
Handles sparse data by storing both values and indices in shared memory
Processes tiles in parallel across thread blocks
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule

# Set CUDA compiler path
os.environ["CUDA_PATH"] = str(Path(subprocess.check_output(["where", "nvcc"], text=True).strip()).parent.parent)


# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, index, num_warmup):
    # Ensure A is in CSR format and B is in CSC format
    A_csr = A.tocsr().astype(np.float32)
    B_csc = B.tocsc().astype(np.float32)

    # Extract CSR components for A
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr

    # Extract CSC components for B
    B_data = B_csc.data
    B_indices = B_csc.indices
    B_indptr = B_csc.indptr

    # Add error checking for GPU memory allocation
    def safe_gpu_alloc(nbytes):
        try:
            mem = cuda.mem_alloc(nbytes)
            return mem
        except cuda.MemoryError:
            raise RuntimeError(f"Failed to allocate {nbytes} bytes on GPU")

    try:
        with nvtx.annotate(f"prepare {index}", domain="claude_pycuda_sparse_tiled"):
            # Update memory allocation with safety checks
            try:
                A_data_gpu = safe_gpu_alloc(A_data.nbytes)
                A_indices_gpu = safe_gpu_alloc(A_indices.nbytes)
                A_indptr_gpu = safe_gpu_alloc(A_indptr.nbytes)
                B_data_gpu = safe_gpu_alloc(B_data.nbytes)
                B_indices_gpu = safe_gpu_alloc(B_indices.nbytes)
                B_indptr_gpu = safe_gpu_alloc(B_indptr.nbytes)
                C_gpu = safe_gpu_alloc(A_csr.shape[0] * B_csc.shape[1] * np.float32().itemsize)
            except RuntimeError as e:
                print(f"GPU memory allocation failed: {e}")
                raise

            # Safe memory transfer
            cuda.memcpy_htod(A_data_gpu, A_data)
            cuda.memcpy_htod(A_indices_gpu, A_indices)
            cuda.memcpy_htod(A_indptr_gpu, A_indptr)
            cuda.memcpy_htod(B_data_gpu, B_data)
            cuda.memcpy_htod(B_indices_gpu, B_indices)
            cuda.memcpy_htod(B_indptr_gpu, B_indptr)

            """
            CUDA implementation of sparse matrix multiplication using CSR format
            
            Binary search function:
            - Finds elements in sparse matrix columns
            - Parameters: array (sorted), left/right indices, target value
            - Returns index if found, -1 otherwise
            
            Sparse matrix multiplication kernel:
            - Multiplies matrices A * B in CSR format
            - Uses 32x32 thread blocks with shared memory
            - Binary search to find matching elements
            - Parallel reduction for final sum
            """

            mod = SourceModule(
                """
            #define TILE_SIZE 32
            #define BLOCK_SIZE 32

            __global__ void sparse_matmul_tiled(
                const float *A_data, const int *A_indices, const int *A_indptr,  // A in CSR
                const float *B_data, const int *B_indices, const int *B_indptr,  // B in CSC
                float *C, int num_rows_A, int num_cols_A, int num_cols_B
            ) {
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
                for(int t = 0; t < num_tiles; t++) {
                    // Clear shared memory tiles
                    A_tile[threadIdx.y][threadIdx.x] = 0.0f;
                    B_tile[threadIdx.y][threadIdx.x] = 0.0f;
                    A_cols[threadIdx.y][threadIdx.x] = -1;
                    B_rows[threadIdx.y][threadIdx.x] = -1;
                    __syncthreads();
                    
                    // Load A tile - each thread loads one element
                    if(row < num_rows_A) {
                        int row_start = A_indptr[row];
                        int row_end = A_indptr[row + 1];
                        
                        for(int i = row_start; i < row_end; i++) {
                            int col_idx = A_indices[i];
                            // Check if element belongs to current tile
                            if(col_idx >= t * TILE_SIZE && col_idx < (t + 1) * TILE_SIZE) {
                                int tile_idx = col_idx - t * TILE_SIZE;
                                A_tile[threadIdx.y][tile_idx] = A_data[i];
                                A_cols[threadIdx.y][tile_idx] = col_idx;
                            }
                        }
                    }
                    
                    // Load B tile - each thread loads one element
                    if(col < num_cols_B) {
                        int col_start = B_indptr[col];
                        int col_end = B_indptr[col + 1];
                        
                        for(int i = col_start; i < col_end; i++) {
                            int row_idx = B_indices[i];
                            // Check if element belongs to current tile
                            if(row_idx >= t * TILE_SIZE && row_idx < (t + 1) * TILE_SIZE) {
                                int tile_idx = row_idx - t * TILE_SIZE;
                                B_tile[tile_idx][threadIdx.x] = B_data[i];
                                B_rows[tile_idx][threadIdx.x] = row_idx;
                            }
                        }
                    }
                    __syncthreads();
                    
                    // Compute partial products for this tile
                    if(row < num_rows_A && col < num_cols_B) {
                        for(int k = 0; k < TILE_SIZE; k++) {
                            if(A_cols[threadIdx.y][k] == B_rows[k][threadIdx.x] && 
                            A_cols[threadIdx.y][k] != -1) {
                                sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
                            }
                        }
                    }
                    __syncthreads();
                }
                
                // Store final result
                if(row < num_rows_A && col < num_cols_B) {
                    C[row * num_cols_B + col] = sum;
                }
            }
            """
            )

            sparse_matmul = mod.get_function("sparse_matmul_tiled")

            block_size = (32, 32, 1)

            # Adjust grid size calculation to ensure coverage
            grid_size = (
                int(np.ceil(B_csc.shape[1] / block_size[0])),
                int(np.ceil(A_csr.shape[0] / block_size[1])),
                1,
            )

            # Ensure block dimensions do not exceed maximum allowed
            block_size = (min(block_size[0], 32), min(block_size[1], 32), 1)

            # Warmup runs
            TILE_SIZE = 32  # Should match the value in the CUDA kernel
            block_size = (TILE_SIZE, TILE_SIZE, 1)  # Fixed for tiled implementation

            # Calculate grid dimensions to cover the entire matrix with tiles
            grid_size = (
                (B_csc.shape[1] + TILE_SIZE - 1) // TILE_SIZE,
                (A_csr.shape[0] + TILE_SIZE - 1) // TILE_SIZE,
                1,
            )

            # Calculate shared memory size
            shared_mem_size = (TILE_SIZE * TILE_SIZE * 4 * 2) + (  # For A_tile and B_tile (float)
                TILE_SIZE * TILE_SIZE * 4 * 2
            )  # For A_cols and B_rows (int)

        # Warmup runs with shared memory
        with nvtx.annotate(f"warmup {index}", domain="claude_pycuda_sparse_tiled"):
            for _ in range(num_warmup):
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    np.int32(A_csr.shape[0]),
                    np.int32(A_csr.shape[1]),
                    np.int32(B_csc.shape[1]),
                    block=block_size,
                    grid=grid_size,
                    shared=shared_mem_size,
                )
                cuda.Context.synchronize()

        # Actual test runs with shared memory
        with nvtx.annotate(f"main {index}", domain="claude_pycuda_sparse_tiled"):
            sparse_matmul(
                A_data_gpu,
                A_indices_gpu,
                A_indptr_gpu,
                B_data_gpu,
                B_indices_gpu,
                B_indptr_gpu,
                C_gpu,
                np.int32(A_csr.shape[0]),
                np.int32(A_csr.shape[1]),
                np.int32(B_csc.shape[1]),
                block=block_size,
                grid=grid_size,
                shared=shared_mem_size,
            )
            cuda.Context.synchronize()

        # Safe memory transfer back
        C_dense = np.empty((A_csr.shape[0], B_csc.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        return C_dense

    finally:
        # Ensure GPU memory is always freed
        try:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()
        except:
            pass


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    num_nodes = graph_info["num_nodes"]
    context = cuda.Device(0).make_context()
    
    # Prepare adjacency matrix
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    adjacency_matrix = adjacency_matrix.tocsr()

    try:
        return sparse_matrix_multiply_pycuda(adjacency_matrix, feature_matrix, index, num_warmup)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()
