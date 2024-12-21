import numpy as np
import pickle
import json
import time
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import networkx as nx
from pathlib import Path

# Print GPU information
def print_gpu_info():
    device = cuda.Device(0)  
    print("GPU Information:")
    print(f"  Name: {device.name()}")
    print(f"  Compute Capability: {device.compute_capability()}")
    print(f"  Total Memory: {device.total_memory() / 1e9:.2f} GB")

# Load graphs
def load_graphs(graph_dir, graph_indices=None):
    graphs = []
    indices_filter = "*" if graph_indices is None else f"[{graph_indices}]"
    for graph_file in Path(graph_dir).glob(f"graph_{indices_filter}.pkl"):
        with open(graph_file, "rb") as f:
            graph_data = pickle.load(f)
            graphs.append(graph_data)
            print(f"Loaded graph: {graph_file}")
    return graphs

# Matrix multiplication method for dense matrices (one thread per output cell)
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

# Run tests and collect results
def process_graphs(graphs):
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

        # Build dense adjacency matrix
        adjacency_matrix = nx.to_numpy_array(graph, dtype=np.float32)

        # Perform multiplication with feature matrix
        start_time = time.time()
        aggregated_feature_matrix = dense_matrix_multiply_pycuda(adjacency_matrix, feature_matrix)
        end_time = time.time()

        elapsed_time = end_time - start_time
        results.append(
            {
                "graph_index": index,
                "graph_name": name,
                "graph_type": graph_type,
                "method": "pycuda_dense",
                "time_seconds": elapsed_time,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_nodes": num_nodes,
                "sparsity": sparsity,
            }
        )
    return results

# Save results
def save_results(results, result_file):
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results have been saved to '{result_file}'.")


if __name__ == "__main__":
    print_gpu_info()  # Print GPU details

    graph_dir = Path(r"<path>")
    result_file = Path(r"<path>")

    # Load graphs
    graphs = load_graphs(graph_dir)
    if not graphs:
        print("No graphs found to process.")
    else:
        # GPU warmup
        st = time.time()
        gpu_warmup()
        et = time.time()
        print(f"GPU warmup time: {et - st:.4f} seconds")

        # Process graphs
        results = process_graphs(graphs)

        # Save results
        save_results(results, result_file)

        print(json.dumps(results, indent=4))
