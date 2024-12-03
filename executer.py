import argparse
import collections.abc
import importlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import scipy.sparse as sp

from utils.verification import verify_result


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


methods = [
    "chatgpt_cupy_sparse",
    "chatgpt_pycuda_decomposed",
    "chatgpt_pycuda_dense",
    "chatgpt_pycuda_sparse",
    "chatgpt_pytorch_dense",
    "chatgpt_pytorch_sparse",
    "claude_pycuda_sparse_csr_csc",
    "claude_pycuda_sparse_instrumented",
    "claude_pycuda_sparse_tiled",
    "claude_pycuda_sparse_tiled_coalesced",
    "claude_pycuda_sparse",
]

# Add command line argument parsing
parser = argparse.ArgumentParser(description="Run GNN experiments with CuPy sparse matrices")
parser.add_argument("--method", type=str, default="claude_pycuda_sparse", choices=methods, help="Method to run")
parser.add_argument("--verify", default=False, action="store_true", help="Enable profiling")
parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
parser.add_argument("--test-runs", type=int, default=1, help="Number of test runs for timing")
parser.add_argument("--graphs", type=str, default=None, help="Index pattern of graphs to process")
args = parser.parse_args()

# import chosen method
method = importlib.import_module(f"scripts.{args.method}")

# Load graphs
graphs = []
graph_indices = "*" if args.graphs is None else f"[{args.graphs}]"
for graph_file in Path("graphs").glob(f"graph_{graph_indices}.pkl"):
    with open(graph_file, "rb") as f:
        graphs.append(pickle.load(f))

results = defaultdict(dict)
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    graph_type = graph_info["type"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    sparsity = graph_info["sparsity"]

    print(f"Testing graph {index}...")

    # Execute the method
    result, mean_time, std_time = method.execute(graph_info, num_warmup=args.warmup, num_runs=args.test_runs)

    # Verify the result
    is_correct = True  # Assume correct by default
    if args.verify:
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=float)
        is_correct = verify_result(result, adjacency_matrix, feature_matrix)

    print(f"Processing completed successfully.")

    results[args.method][index] = {
        "graph_index": index,
        "graph_name": name,
        "graph_type": graph_type,
        "method": args.method,
        "time_seconds": mean_time / 1000.0,  # Convert ms to seconds
        "time_std": std_time / 1000.0,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_nodes": num_nodes,
        "sparsity": sparsity,
        "is_correct": is_correct,
    }

if not Path("gnn_results.json").exists():
    prev_results = {}
else:
    with open("gnn_results.json", "r") as f:
        try:
            prev_results = json.load(f)
        except json.JSONDecodeError:
            prev_results = {}

update_dict(prev_results, results)

with open("gnn_results.json", "w") as f:
    json.dump(prev_results, f, indent=4)

print("\nResults have been saved to 'gnn_results.json'.")
