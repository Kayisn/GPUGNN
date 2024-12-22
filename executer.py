import argparse
import importlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from utils.verification import verify_result

results_path = Path("results") / "results.json"

if __name__ == "__main__":
    # Get all available methods
    methods = [path.stem for path in Path("scripts").glob("*.py") if path.stem != "__init__"]

    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Run GNN experiments with CuPy sparse matrices")
    parser.add_argument("--method", "-m", type=str, choices=methods, help="Method to run", required=True)
    parser.add_argument("--verify", default=False, action="store_true", help="Verify the result")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--graphs", "-g", type=str, default=None, help="Index pattern of graphs to process")
    args = parser.parse_args()

    # import chosen method
    method = importlib.import_module(f"scripts.{args.method}")

    # Load graphs
    graphs = []
    graph_indices = "*" if args.graphs is None or args.graphs == "all" else f"[{args.graphs}]"
    for graph_file in Path("graphs").glob(f"graph_{graph_indices}.pkl"):
        with open(graph_file, "rb") as f:
            graphs.append(pickle.load(f))

    results = defaultdict(dict)
    for graph_info in graphs:
        graph_idx = graph_info["index"]
        graph_name = graph_info["name"]
        graph_type = graph_info["type"]
        graph = graph_info["graph"]
        feature_matrix = graph_info["feature_matrix"]
        num_nodes = graph_info["num_nodes"]
        sparsity = graph_info["sparsity"]

        print(f"Testing graph {graph_idx}...")

        # Execute the method
        result = method.execute(graph_info, num_warmup=args.warmup)

        # Verify the result
        is_correct = None
        if args.verify:
            adjacency_matrix = nx.to_scipy_sparse_array(graph, format="lil", dtype=float)
            is_correct = bool(verify_result(result, adjacency_matrix, feature_matrix))
            print(f"Verification: {'Correct' if is_correct else 'Incorrect'}")

        results[args.method][graph_idx] = {
            "graph_name": graph_name,
            "graph_type": graph_type,
            "method": args.method,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
            "is_correct": is_correct,
        }

        print(f"Processing completed successfully.\n")

    if not results_path.exists():
        prev_results = {}
    else:
        with open(results_path, "r") as f:
            try:
                prev_results = json.load(f)
            except json.JSONDecodeError:
                prev_results = {}

    for method, report in results.items():
        prev_results[method] = prev_results.get(method, {})
        for graph_idx, result in report.items():
            if "metrics" in prev_results[method].get(graph_idx, {}):
                result["metrics"] = prev_results[method][graph_idx]["metrics"]
            prev_results[method][graph_idx] = result

    with open(results_path, "w") as f:
        json.dump(prev_results, f, indent=4)

    print("Results have been saved to 'results.json'.")
