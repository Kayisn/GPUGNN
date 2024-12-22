import argparse
import importlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from utils.verification import verify_result

graph_dir = Path("graphs")
results_path = Path("results") / "results.json"


def module_from_file(file_path: Path):
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_graph_indices(graphs):
    if not graphs or graphs == "all":
        return list(graph_dir.glob("graph_*.pkl"))
    if "," in graphs:
        return [graph_dir / f"graph_{i}.pkl" for i in graphs.split(",")]
    if "-" in graphs:
        start, end = map(int, graphs.split("-"))
        return [graph_dir / f"graph_{i}.pkl" for i in range(start, end + 1)]
    return [graph_dir / f"graph_{graphs}.pkl"]


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Run GNN experiments with CuPy sparse matrices")
    parser.add_argument("--method", "-m", type=str, help="Path of the method to run", required=True)
    parser.add_argument("--verify", default=False, action="store_true", help="Verify the result")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--graphs", "-g", type=str, default=None, help="Index pattern of graphs to process")
    args = parser.parse_args()

    # import chosen method
    method = module_from_file(Path(args.method))

    # Load graphs
    graphs = []
    print(f"Loading graphs...")
    for graph_file in parse_graph_indices(args.graphs):
        with open(graph_file, "rb") as f:
            graphs.append(pickle.load(f))

    results = defaultdict(dict)
    for graph_info in graphs:
        graph_idx = str(graph_info["index"])
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

        results[method.__name__][graph_idx] = {
            "graph_name": graph_name,
            "graph_type": graph_type,
            "method": method.__name__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
            "is_correct": is_correct,
        }

        print(f"Processing completed successfully.\n")

    prev_results = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            try:
                prev_results = json.load(f)
            except json.JSONDecodeError:
                pass

    for method, report in results.items():
        prev_results[method] = prev_results.get(method, {})
        for graph_idx, result in report.items():
            if "metrics" in prev_results[method].get(graph_idx, {}):
                result["metrics"] = prev_results[method][graph_idx]["metrics"]
            prev_results[method][graph_idx] = result

    with open(results_path, "w") as f:
        json.dump(prev_results, f, indent=4)

    print("Results have been saved to 'results.json'.")
