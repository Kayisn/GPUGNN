import json
import pickle
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    graph_type = graph_info["type"]
    graph = graph_info["graph"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]
    feature_matrix = torch.tensor(
        graph_info["feature_matrix"].toarray(), dtype=torch.float32
    ).to(device)
    print(f"Testing graph {index}")
    # Prepare adjacency matrix
    num_nodes = feature_matrix.shape[0]
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32).to(device)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0

    # convert to sparse matrix
    adjacency_matrix = adjacency_matrix.to_sparse()

    # convert feature matrix to sparse matrix
    feature_matrix = feature_matrix.to_sparse()

    # Perform forward pass and measure time
    memory_idle = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = time.time()
    output = torch.sparse.mm(adjacency_matrix, feature_matrix)
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_allocated = (torch.cuda.max_memory_allocated(device) - memory_idle) / 1024**2

    results.append(
        {
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "pytorch_sparse",
            "time_seconds": elapsed_time,
            "memory_peak_mb": memory_allocated,
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
