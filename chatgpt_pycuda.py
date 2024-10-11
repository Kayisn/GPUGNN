import numpy as np
import scipy.sparse as sp
import pickle
import json
import time
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import networkx as nx

# Load graphs
with open('gnn_test_graphs_with_features.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Define the PyCUDA-based multiplication method
def dense_matrix_multiply_pycuda(A, B):
    A_dense = A.toarray().astype(np.float32) if sp.issparse(A) else A.astype(np.float32)
    B_dense = B.astype(np.float32)

    A_gpu = cuda.mem_alloc(A_dense.nbytes)
    B_gpu = cuda.mem_alloc(B_dense.nbytes)
    C_gpu = cuda.mem_alloc(A_dense.shape[0] * B_dense.shape[1] * A_dense.dtype.itemsize)

    cuda.memcpy_htod(A_gpu, A_dense)
    cuda.memcpy_htod(B_gpu, B_dense)

    mod = SourceModule("""
    __global__ void matmul(float *A, float *B, float *C, int widthA, int widthB) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < widthA && col < widthB) {
            float sum = 0;
            for (int k = 0; k < widthA; ++k) {
                sum += A[row * widthA + k] * B[k * widthB + col];
            }
            C[row * widthB + col] = sum;
        }
    }
    """)

    matmul = mod.get_function("matmul")
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(B_dense.shape[1] / 16)), int(np.ceil(A_dense.shape[0] / 16)), 1)
    matmul(A_gpu, B_gpu, C_gpu, np.int32(A_dense.shape[1]), np.int32(B_dense.shape[1]), block=block_size, grid=grid_size)

    C_dense = np.empty((A_dense.shape[0], B_dense.shape[1]), dtype=np.float32)
    cuda.memcpy_dtoh(C_dense, C_gpu)

    return C_dense

# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info['index']
    name = graph_info['name']
    graph_type = graph_info['type']
    graph = graph_info['graph']
    feature_matrix = graph_info['feature_matrix']
    num_nodes = graph_info['num_nodes']
    sparsity = graph_info['sparsity']
    print(f"Testing graph {index}")
    
    # Perform multiplication (example using BFS and feature matrix)
    aggregated_feature_matrix = feature_matrix.copy()
    start_time = time.time()

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if neighbors:
            neighbor_features = sp.vstack([feature_matrix.getrow(neighbor) for neighbor in neighbors])
            aggregated_feature_matrix[node] = dense_matrix_multiply_pycuda(neighbor_features.T, np.ones((neighbor_features.shape[0], 1))).T

    end_time = time.time()
    elapsed_time = end_time - start_time
    results.append({
        'graph_index': index,
        'graph_name': name,
        'graph_type': graph_type,
        'method': 'pycuda_dense',
        'time_seconds': elapsed_time,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_nodes': num_nodes,
        'sparsity': sparsity
    })

import os
# Load existing results or create a new one
if os.path.exists('gnn_results.json'):
    with open('gnn_results.json', 'r') as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []  # Initialize as an empty list if the file is empty or corrupted
else:
    all_results = []

# Update results by replacing existing ones by graph index and method
for result in results:
    # Check if the result already exists in the list
    if any(r['graph_index'] == result['graph_index'] and r['method'] == result['method'] for r in all_results):
        # If so, replace the existing result
        all_results = [r for r in all_results if not (r['graph_index'] == result['graph_index'] and r['method'] == result['method'])]
        all_results.append(result)
    else:
        all_results.append(result)

# Save results
with open('gnn_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)

# Print confirmation
print("Results have been saved to 'gnn_results.json'.")