import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
from contextlib import contextmanager
from filelock import FileLock  # Need to add this dependency

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule
from occupancy_tracker import OccupancyTracker
from verification import verify_result

# Add command line argument parsing
parser = argparse.ArgumentParser(description='CUDA Sparse Matrix Multiplication with optional profiling')
parser.add_argument('--profile', action='store_true', help='Enable occupancy profiling', default=False)
parser.add_argument('--block-size', type=int, nargs=2, default=[32, 32], 
                    help='Block size dimensions (x y), default: 32 32')
parser.add_argument('--warmup', type=int, default=2,
                    help='Number of warmup runs')
parser.add_argument('--test-runs', type=int, default=5,
                    help='Number of test runs for timing')
args = parser.parse_args()

# Set CUDA compiler path before importing pycuda
os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

# Add a lock for CUDA context
cuda_context_lock = threading.Lock()

@contextmanager
def cuda_context():
    """Thread-safe CUDA context manager"""
    with cuda_context_lock:
        context = cuda.Device(0).make_context()
        try:
            yield context
        finally:
            context.pop()

# Memory monitor now part of profiling
def memory_monitor(duration_seconds):
    """Measure peak memory usage over specified duration"""
    peak_memory_usage = 0
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = total_mem - free_mem
        peak_memory_usage = max(peak_memory_usage, used_mem)
        time.sleep(0.1)
    return peak_memory_usage

# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, num_warmup=2, num_test_runs=5, profile_mode=False):
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

        print("Running sparse matrix multiplication on GPU...")

        mod = SourceModule("""
        __global__ void sparse_matmul(
            const float *A_data, const int *A_indices, const int *A_indptr,  // A in CSR
            const float *B_data, const int *B_indices, const int *B_indptr,  // B in CSC
            float *C, int num_rows_A, int num_cols_A, int num_cols_B
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

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
                C[row * num_cols_B + col] = sum;
            }
        }
        """)

        print("Kernel compilation successful.")

        sparse_matmul = mod.get_function("sparse_matmul")

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
        block_size = (
            min(block_size[0], 32),
            min(block_size[1], 32),
            1
        )
        
        if profile_mode:
            tracker.log_statistics(sparse_matmul, block_size, grid_size)

        # Warmup runs
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, 
                np.int32(A_csr.shape[0]),  # num_rows_A
                np.int32(A_csr.shape[1]),  # num_cols_A
                np.int32(B_csc.shape[1]),  # num_cols_B
                block=block_size,
                grid=grid_size
            )
            cuda.Context.synchronize()

        # Actual test runs with timing
        times = []
        for _ in range(num_test_runs):
            start = cuda.Event()
            end = cuda.Event()
            
            start.record()
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, 
                np.int32(A_csr.shape[0]),  # num_rows_A
                np.int32(A_csr.shape[1]),  # num_cols_A
                np.int32(B_csc.shape[1]),  # num_cols_B
                block=block_size,
                grid=grid_size
            )
            end.record()
            end.synchronize()
            
            times.append(start.time_till(end))

        # Safe memory transfer back
        C_dense = np.empty((A_csr.shape[0], B_csc.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)
        
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
        except:
            pass


# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    graph_type = graph_info["type"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]
    print(f"Testing graph {index}")

    # Setup memory tracking with proper thread safety
    free_mem, total_mem = cuda.mem_get_info()
    memory_idle = total_mem - free_mem
    stop_event = threading.Event()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        if args.profile:
            memory_thread = executor.submit(memory_monitor, stop_event)
        
        try:
            with cuda_context() as ctx:
                # Create matrices
                adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
                for node in graph.nodes:
                    for neighbor in graph.neighbors(node):
                        adjacency_matrix[node, neighbor] = 1.0
                adjacency_matrix = adjacency_matrix.tocsr()
                feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

                # Profile mode includes memory monitoring and occupancy tracking
                memory_usage = 0
                if args.profile:
                    tracker = OccupancyTracker()
                    free_mem_initial, total_mem = cuda.mem_get_info()
                    
                    # Run with profiling
                    _, _, _ = sparse_matrix_multiply_pycuda(
                        adjacency_matrix,
                        feature_matrix,
                        num_warmup=1,
                        num_test_runs=1,
                        profile_mode=True
                    )
                    
                    # Quick memory measurement during a test run
                    memory_usage = memory_monitor(2.0) - (total_mem - free_mem_initial)
                    memory_usage = memory_usage / 1024**2  # Convert to MB
                    print(f"Peak memory usage: {memory_usage:.2f} MB")

                # Actual benchmarking run
                result, avg_time, std_time = sparse_matrix_multiply_pycuda(
                    adjacency_matrix,
                    feature_matrix,
                    num_warmup=args.warmup,
                    num_test_runs=args.test_runs,
                    profile_mode=False
                )

                is_correct = verify_result(result, adjacency_matrix, feature_matrix)

                results.append({
                    "graph_index": index,
                    "graph_name": name,
                    "graph_type": graph_type,
                    "method": "pycuda_sparse_csr_csc",
                    "time_seconds": avg_time / 1000.0,  # Convert ms to seconds
                    "time_std": std_time / 1000.0,
                    "memory_peak_mb": memory_usage if args.profile else None,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_nodes": num_nodes,
                    "sparsity": sparsity,
                    "is_correct": is_correct
                })

                if not is_correct:
                    print(f"Graph {name} failed verification.")

        except cuda.LaunchError as e:
            print(f"CUDA launch failed: {e}")
            stop_event.set()
            continue
        except Exception as e:
            print(f"Error processing graph {name}: {e}")
            stop_event.set()
            continue

# Thread-safe file operations
def save_results(new_results):
    lock_path = "gnn_results.json.lock"
    with FileLock(lock_path):
        if os.path.exists("gnn_results.json"):
            with open("gnn_results.json", "r") as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []

        # Update results
        for result in new_results:
            all_results = [r for r in all_results 
                         if not (r["graph_index"] == result["graph_index"] 
                               and r["method"] == result["method"])]
            all_results.append(result)

        with open("gnn_results.json", "w") as f:
            json.dump(all_results, f, indent=4)

# Save results thread-safely
save_results(results)

# Print confirmation
print("Results have been saved to 'gnn_results.json'.")