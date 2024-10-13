import json
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)


# Memory tracking thread function
def memory_monitor(stop_event, interval=0.1):
    peak_memory_usage = 0
    while not stop_event.is_set():
        used_mem = cp.get_default_memory_pool().used_bytes()
        peak_memory_usage = max(peak_memory_usage, used_mem)
        time.sleep(interval)
    return peak_memory_usage


# Define the cuSPARSE-based multiplication method
def sparse_matrix_multiply_cusparse_gpu(A, B):
    A_csr = cp.sparse.csr_matrix(A)
    B_csr = cp.sparse.csr_matrix(B)
    C_csr = A_csr.dot(B_csr)
    return C_csr


# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    type = graph_info["type"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]

    print(f"Testing graph {index}")

    memory_idle = cp.get_default_memory_pool().used_bytes()
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    memory_thread = executor.submit(memory_monitor, stop_event)
    start_time = time.time()

    # Convert feature matrix to CuPy for GPU operations
    feature_matrix_gpu = cp.sparse.csr_matrix(feature_matrix)

    # Initialize an empty aggregated feature matrix
    aggregated_feature_matrix_gpu = feature_matrix_gpu.copy()

    # Perform aggregation using neighbors
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 0:
            # Extract neighbors' features as a single sparse matrix
            neighbor_features_gpu = feature_matrix_gpu[neighbors, :]
            # Aggregate neighbor features using cuSPARSE multiplication
            aggregated_value = sparse_matrix_multiply_cusparse_gpu(
                neighbor_features_gpu.T, cp.ones((len(neighbors), 1), dtype=cp.float32)
            ).T
            # Update the aggregated feature matrix for the current node
            aggregated_feature_matrix_gpu[node, :] = aggregated_value

    end_time = time.time()
    stop_event.set()
    elapsed_time = end_time - start_time
    peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2

    # Convert aggregated matrix back to host for storing results if needed
    results.append(
        {
            "graph_index": index,
            "graph_name": name,
            "graph_type": type,
            "method": "cupy_sparse",
            "time_seconds": elapsed_time,
            "memory_peak_mb": peak_memory_usage,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
        }
    )

# Load existing results or create a new one
import os

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
