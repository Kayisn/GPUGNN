import json
import pickle
import time
import numpy as np
import scipy.sparse as sp
import sparse_matrix_multiply
import sys

# Get the file name from command line arguments or use the default
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "gnn_test_graphs_with_features.pkl"

# Load graphs
with open(input_file, "rb") as f:
    graphs = pickle.load(f)


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

    # Convert adjacency matrix to CSR format
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    adjacency_matrix = adjacency_matrix.tocsr()

    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    # Perform sparse matrix multiplication using C++/CUDA
    A_data = adjacency_matrix.data
    A_indices = adjacency_matrix.indices
    A_indptr = adjacency_matrix.indptr
    B_data = feature_matrix.data
    B_indices = feature_matrix.indices
    B_indptr = feature_matrix.indptr

    result, elapsed_time, peak_memory_usage = sparse_matrix_multiply.sparse_matrix_multiply(
        A_data, A_indices, A_indptr, B_data, B_indices, B_indptr, num_nodes, num_nodes, feature_matrix.shape[1]
    )

    results.append(
        {
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "c++_cuda_sparse",
            "time_seconds": elapsed_time,
            "memory_peak_mb": peak_memory_usage,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
        }
    )

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