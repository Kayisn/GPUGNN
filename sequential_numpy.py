import json
import pickle
import threading
import time

import numpy as np
import scipy.sparse as sp

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)


# Define the PyCUDA-based multiplication method
def matrix_multiply(A, B):
    # Ensure A is in CSR format for efficient multiplication
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    if not sp.isspmatrix_csr(B):
        B = sp.csr_matrix(B)
    return A @ B



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


    # Convert adjacency matrix to sparse format
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    
    # Convert to CSR format for efficient multiplication
    adjacency_matrix = adjacency_matrix.tocsr()
    feature_matrix = sp.csr_matrix(feature_matrix)

    start_time = time.time()
    result = matrix_multiply(adjacency_matrix, feature_matrix)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
 
    results.append(
        {
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "sequential_numpy",
            "time_seconds": elapsed_time,
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
