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


# Define the dense layer for GNN-style aggregation
class DenseGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseGNNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        aggregated = torch.matmul(adjacency_matrix, x)
        out = self.fc(aggregated)
        return out


# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    type = graph_info["type"]
    graph = graph_info["graph"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]
    feature_matrix = torch.tensor(
        graph_info["feature_matrix"].toarray(), dtype=torch.float32
    )
    print(f"Testing graph {index}")
    # Prepare adjacency matrix
    num_nodes = feature_matrix.shape[0]
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0

    # Initialize model
    model = DenseGNNLayer(in_features=10, out_features=10)

    # Perform forward pass and measure time
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = model(feature_matrix, adjacency_matrix)
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2

    results.append(
        {
            "graph_index": index,
            "graph_name": name,
            "graph_type": type,
            "method": "pytorch_dense",
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
