import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import nvtx
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule

from utils.occupancy_tracker import OccupancyTracker

# Set CUDA compiler path
cuda_path = Path(
    subprocess.run(["where", "nvcc"], capture_output=True).stdout.decode().split("\n")[0].strip()
).parent.parent
os.environ["CUDA_PATH"] = str(cuda_path)

# os.environ["PATH"] = (
#     r"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64"
#     + os.pathsep
#     + os.environ["PATH"]
# )


@dataclass
class NodeWorkload:
    """Track computation and communication patterns per node"""

    input_edges: int
    output_edges: int
    computation_time: float
    neighbors: List[int]


# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, index, num_warmup=2, num_test_runs=5, profile_mode=False):
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
        with nvtx.annotate(f"prepare {index}", domain="claude_pycuda_sparse_instrumented"):
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
            __global__ void sparse_matmul_instrumented(
                const float *A_data, const int *A_indices, const int *A_indptr,  // A in CSR
                const float *B_data, const int *B_indices, const int *B_indptr,  // B in CSC
                float *C, int *work_counters, // Track operations per thread
                int num_rows_A, int num_cols_A, int num_cols_B
            ) {
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                int tid = row * num_cols_B + col;
                int ops_count = 0;

                if(row < num_rows_A && col < num_cols_B) {
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
                    while(a_idx < row_end && b_idx < col_end) {
                        ops_count++;
                        int a_col = A_indices[a_idx];    // Column index in A
                        int b_row = B_indices[b_idx];    // Row index in B
                        
                        if(a_col == b_row) {
                            // Matching indices - multiply and add
                            sum += A_data[a_idx] * B_data[b_idx];
                            a_idx++;
                            b_idx++;
                        }
                        else if(a_col < b_row) {
                            // Need to move forward in A
                            a_idx++;
                        }
                        else {
                            // Need to move forward in B
                            b_idx++;
                        }
                    }
                    
                    // Store result - use row-major ordering since output is dense
                    C[tid] = sum;
                    work_counters[tid] = ops_count;
                }
            }
            """
            )

            sparse_matmul = mod.get_function("sparse_matmul_instrumented")

            # Only track occupancy in profile mode
            if profile_mode:
                tracker = OccupancyTracker()
                suggested_block_size = tracker.suggest_block_size(sparse_matmul)
                print("\nProfiling Information:")
                print(f"Suggested block size: {suggested_block_size}")
                block_size = suggested_block_size  # Use suggested size when profiling
            else:
                block_size = (32, 32, 1)  # Default size otherwise

            # Adjust grid size calculation to ensure coverage
            grid_size = (
                int(np.ceil(B_csc.shape[1] / block_size[0])),
                int(np.ceil(A_csr.shape[0] / block_size[1])),
                1,
            )

            # Ensure block dimensions do not exceed maximum allowed
            block_size = (min(block_size[0], 32), min(block_size[1], 32), 1)

            # Allocate counter array
            work_counters = np.zeros((A_csr.shape[0] * B_csc.shape[1]), dtype=np.int32)
            work_counters_gpu = safe_gpu_alloc(work_counters.nbytes)

        # Warmup runs
        with nvtx.annotate(f"warmup {index}", domain="claude_pycuda_sparse_instrumented"):
            for _ in range(num_warmup):
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    work_counters_gpu,
                    np.int32(A_csr.shape[0]),  # num_rows_A
                    np.int32(A_csr.shape[1]),  # num_cols_A
                    np.int32(B_csc.shape[1]),  # num_cols_B
                    block=block_size,
                    grid=grid_size,
                )
                cuda.Context.synchronize()

        # Actual test runs with timing
        with nvtx.annotate(f"main {index}", domain="claude_pycuda_sparse_instrumented"):
            times = []
            for _ in range(num_test_runs):
                start = cuda.Event()
                end = cuda.Event()

                start.record()
                sparse_matmul(
                    A_data_gpu,
                    A_indices_gpu,
                    A_indptr_gpu,
                    B_data_gpu,
                    B_indices_gpu,
                    B_indptr_gpu,
                    C_gpu,
                    work_counters_gpu,
                    np.int32(A_csr.shape[0]),  # num_rows_A
                    np.int32(A_csr.shape[1]),  # num_cols_A
                    np.int32(B_csc.shape[1]),  # num_cols_B
                    block=block_size,
                    grid=grid_size,
                )
                end.record()
                end.synchronize()

                times.append(start.time_till(end))

        # Safe memory transfer back
        C_dense = np.empty((A_csr.shape[0], B_csc.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        # Analyze workload distribution
        cuda.memcpy_dtoh(work_counters, work_counters_gpu)
        node_workloads: Dict[int, NodeWorkload] = {}
        for row in range(A_csr.shape[0]):
            node_workloads[row] = NodeWorkload(
                input_edges=A_indptr[row + 1] - A_indptr[row],
                output_edges=sum(work_counters[row * B_csc.shape[1] : (row + 1) * B_csc.shape[1]]),
                computation_time=np.mean(times),
                neighbors=list(A_indices[A_indptr[row] : A_indptr[row + 1]]),
            )

        return C_dense, np.mean(times), np.std(times)

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
            work_counters_gpu.free()
        except:
            pass


def execute(graph_info, num_warmup=1, num_runs=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    num_nodes = graph_info["num_nodes"]
    context = cuda.Device(0).make_context()
    try:
        # Create matrices
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0
        adjacency_matrix = adjacency_matrix.tocsr()

        # Actual benchmarking run
        return sparse_matrix_multiply_pycuda(
            adjacency_matrix,
            feature_matrix,
            index,
            num_warmup=num_warmup,
            num_test_runs=num_runs,
            profile_mode=False,
        )
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        context.pop()