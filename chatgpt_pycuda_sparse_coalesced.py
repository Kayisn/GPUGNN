import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule

from verification import verify_result

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

import os
# Set CUDA compiler path before importing pycuda
os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']


# Memory tracking thread function
def memory_monitor(stop_event, context):
    peak_memory_usage = 0
    context.push()  # Push the context to the current thread
    while not stop_event.is_set():
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = total_mem - free_mem
        peak_memory_usage = max(peak_memory_usage, used_mem)
        time.sleep(0.1)  # Sleep for a short duration to avoid busy-waiting
    context.pop()  # Pop the context from the current thread
    return peak_memory_usage

# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, num_warmup=2, num_test_runs=5):
    # Ensure A and B are in CSR format
    A_csr = A.tocsr().astype(np.float32)
    B_csr = B.tocsr().astype(np.float32)

    # Extract CSR components
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr

    B_data = B_csr.data
    B_indices = B_csr.indices
    B_indptr = B_csr.indptr

    # Allocate GPU memory for CSR components
    A_data_gpu = cuda.mem_alloc(A_data.nbytes)
    A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
    A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
    B_data_gpu = cuda.mem_alloc(B_data.nbytes)
    B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
    B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
    C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

    # Copy data to GPU
    cuda.memcpy_htod(A_data_gpu, A_data)
    cuda.memcpy_htod(A_indices_gpu, A_indices)
    cuda.memcpy_htod(A_indptr_gpu, A_indptr)
    cuda.memcpy_htod(B_data_gpu, B_data)
    cuda.memcpy_htod(B_indices_gpu, B_indices)
    cuda.memcpy_htod(B_indptr_gpu, B_indptr)

    # CUDA kernel for sparse matrix multiplication
    mod = SourceModule(
        """
    __global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, float *B_data, int *B_indices, int *B_indptr, float *C, int num_rows, int num_cols, int num_cols_B) {
        extern __shared__ float shared_B_data[];
        
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Load B_data into shared memory for coalesced access
        for (int i = threadIdx.x; i < num_cols_B; i += blockDim.x) {
            if (i < num_cols_B) {
                shared_B_data[i] = B_data[i];  // Coalesce memory access for B_data
            }
        }
        __syncthreads();  // Ensure all threads have loaded data into shared memory

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
                        sum += A_data[idx] * shared_B_data[jdx];
                        break;
                    }
                }
            }
            C[row * num_cols_B + col] = sum;
        }
    }
    """
    )

    sparse_matmul = mod.get_function("sparse_matmul")
    block_size = (16, 16, 1)
    grid_size = (
        int(np.ceil(B_csr.shape[1] / 16)),
        int(np.ceil(A_csr.shape[0] / 16)),
        1,
    )

    try:
        # Warmup runs
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
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
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            end.record()
            end.synchronize()
            
            elapsed_time = start.time_till(end)
            times.append(elapsed_time)

        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Copy the result back to host
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        # Free GPU memory
        A_data_gpu.free()
        A_indices_gpu.free()
        A_indptr_gpu.free()
        B_data_gpu.free()
        B_indices_gpu.free()
        B_indptr_gpu.free()
        C_gpu.free()

        return C_dense, mean_time, std_time

    except:
        # Cleanup on error
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
        raise


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

    # Perform multiplication (example using BFS and feature matrix)
    aggregated_feature_matrix = feature_matrix.copy()

    free_mem, total_mem = cuda.mem_get_info()
    memory_idle = total_mem - free_mem
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    context = cuda.Device(0).make_context()

    memory_thread = executor.submit(memory_monitor, stop_event, context)


    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    adjacency_matrix = adjacency_matrix.tocsr()

    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    time.sleep(0.5)  # Wait for memory thread to start

    try:
        # Execute computation
        result, mean_time, std_time = sparse_matrix_multiply_pycuda(
            adjacency_matrix, 
            feature_matrix,
            num_warmup=2,
            num_test_runs=5
        )

        is_correct = verify_result(result, adjacency_matrix, feature_matrix)

        if not is_correct:
            print(f"Graph {name} failed verification.")
        
        # Stop memory tracking and get results
        stop_event.set()
        peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2

        results.append({
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "pycuda_sparse_gpt_coalesced",
            "time_seconds": mean_time / 1000.0,  # Convert ms to seconds
            "time_std": std_time / 1000.0,  # Convert ms to seconds
            "memory_peak_mb": peak_memory_usage,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
            "is_correct": is_correct
        })

    except cuda.LaunchError as e:
        print(f"CUDA launch failed: {e}")
        stop_event.set()
        continue
    except Exception as e:
        print(f"Error processing graph {name}: {e}")
        stop_event.set()
        continue
    finally:
        context.pop()

import os

# Load existing results or create a new one
if os.path.exists("gnn_results.json"):
    with open("gnn_results.json", "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            # Initialize as an empty list if the file is empty or corrupted
            all_results = []
else:
    all_results = []

# Update results by replacing existing ones by graph index and method
for result in results:
    # Check if the result already exists in the list
    if any(
        r["graph_index"] == result["graph_index"] and r["method"] == result["method"]
        for r in all_results
    ):
        # If so, replace the existing result
        all_results = [
            r
            for r in all_results
            if not (
                r["graph_index"] == result["graph_index"]
                and r["method"] == result["method"]
            )
        ]
        all_results.append(result)
    else:
        all_results.append(result)

# Save results
with open("gnn_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Print confirmation
print("Results have been saved to 'gnn_results.json'.")
