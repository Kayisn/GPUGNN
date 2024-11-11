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

def sparse_matrix_multiply_pytorch(adj_matrix, feat_matrix, num_warmup=2, num_runs=5):
    # Convert scipy sparse matrix to PyTorch sparse tensor
    if not isinstance(adj_matrix, sp.csr_matrix):
        adj_matrix = adj_matrix.to_sparse_csr()
        adj_matrix = sp.csr_matrix((adj_matrix.values().cpu().numpy(),
                                  adj_matrix.indices().cpu().numpy(),
                                  adj_matrix.crow_indices().cpu().numpy()),
                                 shape=adj_matrix.shape)
    
    if not isinstance(feat_matrix, sp.csr_matrix):
        feat_matrix = feat_matrix.to_sparse_csr()
        feat_matrix = sp.csr_matrix((feat_matrix.values().cpu().numpy(),
                                   feat_matrix.indices().cpu().numpy(),
                                   feat_matrix.crow_indices().cpu().numpy()),
                                  shape=feat_matrix.shape)

    # Convert to PyTorch sparse tensors
    adj_sparse = torch.sparse_csr_tensor(
        torch.from_numpy(adj_matrix.indptr).to(device),
        torch.from_numpy(adj_matrix.indices).to(device),
        torch.from_numpy(adj_matrix.data).to(device),
        size=adj_matrix.shape
    )
    
    feat_sparse = torch.sparse_csr_tensor(
        torch.from_numpy(feat_matrix.indptr).to(device),
        torch.from_numpy(feat_matrix.indices).to(device),
        torch.from_numpy(feat_matrix.data).to(device),
        size=feat_matrix.shape
    )

    # Warmup
    for _ in range(num_warmup):
        result = torch.sparse.mm(adj_sparse, feat_sparse)
        torch.cuda.synchronize()

    # Actual runs with timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = torch.sparse.mm(adj_sparse, feat_sparse)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        times.append(elapsed_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    return result, mean_time, std_time

# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    type = graph_info["type"]
    graph = graph_info["graph"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]
    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
    
    print(f"Testing graph {index}")
    
    # Prepare adjacency matrix
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    adjacency_matrix = adjacency_matrix.tocsr()

    # Measure memory and time
    torch.cuda.reset_peak_memory_stats(device)
    memory_idle = torch.cuda.memory_allocated(device)
    
    try:
        output, mean_time, std_time = sparse_matrix_multiply_pytorch(
            adjacency_matrix, 
            feature_matrix,
            num_warmup=2,
            num_runs=5
        )
        
        memory_allocated = (torch.cuda.max_memory_allocated(device) - memory_idle) / 1024**2

        results.append({
            "graph_index": index,
            "graph_name": name,
            "graph_type": type,
            "method": "pytorch_sparse",
            "time_seconds": mean_time / 1000.0,  # Convert ms to seconds
            "time_std": std_time / 1000.0,  # Convert ms to seconds
            "memory_peak_mb": memory_allocated,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
        })
        
    except Exception as e:
        print(f"Error processing graph {name}: {e}")
        continue
    finally:
        # Clear cache
        torch.cuda.empty_cache()

# Load existing results or create new ones
import os

if os.path.exists("gnn_results.json"):
    with open("gnn_results.json", "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []
else:
    all_results = []

# Update results
for result in results:
    if any(r["graph_index"] == result["graph_index"] and 
           r["method"] == result["method"] for r in all_results):
        all_results = [r for r in all_results if not (
            r["graph_index"] == result["graph_index"] and 
            r["method"] == result["method"]
        )]
        all_results.append(result)
    else:
        all_results.append(result)

# Save results
with open("gnn_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Results have been saved to 'gnn_results.json'.")
