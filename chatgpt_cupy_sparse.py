import json
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import scipy.sparse as sp
from cupyx.scipy import sparse as cusp


# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)


# Memory tracking thread function
def memory_monitor(stop_event, interval=0.1):
    peak_memory_usage = 0
    while not stop_event.is_set():
        mem_pool = cp.get_default_memory_pool()
        used_mem = mem_pool.used_bytes()
        peak_memory_usage = max(peak_memory_usage, used_mem)
        time.sleep(interval)
    return peak_memory_usage


# Define the CuPy sparse matrix multiplication method
def sparse_matrix_multiply_cusparse(adj_matrix, feat_matrix, num_warmup=2, num_runs=5):
    # Convert to CSR format and move to GPU
    adj_gpu = cusp.csr_matrix(adj_matrix)
    feat_gpu = cusp.csr_matrix(feat_matrix)

    # Ensure matrices are contiguous and optimally laid out
    adj_gpu.sort_indices()
    feat_gpu.sort_indices()

    # Warmup
    for _ in range(num_warmup):
        result = adj_gpu.dot(feat_gpu)
        cp.cuda.stream.get_current_stream().synchronize()

    # Actual runs with timing
    times = []
    for _ in range(num_runs):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        result = adj_gpu.dot(feat_gpu)
        end_event.record()
        end_event.synchronize()
        
        elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
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
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]

    print(f"Testing graph {index}")

    # Clear memory pool before starting
    cp.get_default_memory_pool().free_all_blocks()
    memory_idle = cp.get_default_memory_pool().used_bytes()
    
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    memory_thread = executor.submit(memory_monitor, stop_event)

    try:
        # Convert feature matrix to sparse format
        feature_matrix_sparse = sp.csr_matrix(feature_matrix)

        # Prepare adjacency matrix
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0
        adjacency_matrix = adjacency_matrix.tocsr()

        # Execute computation
        start_time = time.time()
        result, mean_time, std_time = sparse_matrix_multiply_cusparse(
            adjacency_matrix, feature_matrix_sparse
        )
        end_time = time.time()
        
        # Stop memory tracking and get results
        stop_event.set()
        peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2

        results.append({
            "graph_index": index,
            "graph_name": name,
            "graph_type": type,
            "method": "cupy_sparse",
            "time_seconds": mean_time / 1000.0,  # Convert ms to seconds
            "time_std": std_time / 1000.0,  # Convert ms to seconds
            "memory_peak_mb": peak_memory_usage,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
        })

    except Exception as e:
        print(f"Error processing graph {name}: {e}")
        stop_event.set()
        continue
    finally:
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()

# Load existing results or create a new one
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