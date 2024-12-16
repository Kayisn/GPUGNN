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
import networkx as nx
import requests
import os
import gzip
import pandas as pd

from verification import verify_result

def download_snap_graph(url, filename):
    """
    Downloads and extracts a SNAP graph file.
    """
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.raw.read())
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {url}")
    else:
        print(f"File already exists: {filename}")

def load_snap_graph(filename):
    """
    Loads the Bitcoin Alpha SNAP graph file using NetworkX.
    The file format is expected to be a CSV with source, target, and weight.
    """
    # Read the CSV file
    with gzip.open(filename, 'rt') as f:
        df = pd.read_csv(f, comment='#', header=None)
        df.columns = ['source', 'target', 'weight']
    
    # Create a directed graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    return G

# Update SNAP URL and filename
snap_url = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
snap_filename = "soc-sign-bitcoinalpha.csv.gz"

# Download the graph
download_snap_graph(snap_url, snap_filename)

# Load the graph
G = load_snap_graph(snap_filename)
print(f"Loaded SNAP graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Convert the graph to a sparse adjacency matrix
adjacency_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
num_nodes = G.number_of_nodes()
feature_matrix = sp.identity(num_nodes, format='csr', dtype=np.float32)  # Example feature matrix (identity)

# Memory tracking and PyCUDA-based matrix multiplication
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

# Run a test with the loaded SNAP graph
print(f"Running test with SNAP graph ({snap_filename})")
result, mean_time, std_time = sparse_matrix_multiply_pycuda(
    adjacency_matrix,
    feature_matrix,
    num_warmup=2,
    num_test_runs=5
)
print(f"Mean time: {mean_time / 1000:.4f} seconds, Std time: {std_time / 1000:.4f} seconds")
