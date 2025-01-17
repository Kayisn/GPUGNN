import argparse
import gc
import importlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse as sp

from utils.verification import verify_result

graph_dir = Path("graphs")
snap_dir = graph_dir / "snap"
synthetic_dir = graph_dir / "synthetic"
ssmatrices_dir = graph_dir / "ssmatrices"
results_path = Path("results") / "results.json"


def module_from_file(file_path: Path):
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_synthetic_pattern(graphs_pattern):
    if graphs_pattern is None:
        return []
    if graphs_pattern == "all":
        return synthetic_dir.glob("graph_*.pkl")
    if "," in graphs_pattern:
        return [synthetic_dir / f"graph_{i}.pkl" for i in graphs_pattern.split(",")]
    if "-" in graphs_pattern:
        start, end = map(int, graphs_pattern.split("-"))
        return [synthetic_dir / f"graph_{i}.pkl" for i in range(start, end + 1)]
    return [synthetic_dir / f"graph_{graphs_pattern}.pkl"]


def parse_matrices_pattern(matrices_pattern):
    if matrices_pattern is None:
        return []
    all_matrices = ssmatrices_dir.glob("*.mtx")
    matrices = []
    if matrices_pattern != "all":
        for pattern in matrices_pattern.split(","):
            matrices.extend(filter(lambda m: pattern in m.stem, all_matrices))
    return [Path(m) for m in matrices]


def parse_snap_pattern(networks_pattern):
    if networks_pattern is None:
        return []
    all_networks = snap_dir.glob("*.gz")
    networks = []
    if networks_pattern != "all":
        for pattern in networks_pattern.split(","):
            networks.extend(filter(lambda n: pattern in n.stem, all_networks))
    return [Path(n) for n in networks]


def load_synthetic(graphs_file):
    with open(graphs_file, "rb") as f:
        return pickle.load(f)


def load_matrix(matrix_file):
    matrix = scipy.io.mmread(matrix_file).astype(np.float32)
    num_nodes = matrix.shape[0]
    sparsity = 1 - (matrix.nnz / (num_nodes**2))
    return {
        "index": matrix_file.stem,
        "name": f"{num_nodes}_p_{sparsity:.2f}_{matrix_file.stem}",
        "type": "matrix",
        "graph": nx.from_scipy_sparse_array(matrix),
        "feature_matrix": matrix,
        "num_nodes": num_nodes,
        "sparsity": sparsity,
    }


def load_snap(snap_file):
    if ".txt" in snap_file.suffixes:
        graph = nx.read_edgelist(snap_file, nodetype=int)
    elif ".csv" in snap_file.suffixes:
        graph = nx.read_edgelist(snap_file, delimiter=",", nodetype=int, data=(("source", int), ("target", int)))

    num_nodes = graph.number_of_nodes()
    sparsity = 1 - (graph.number_of_edges() / (num_nodes**2))
    return {
        "index": snap_file.stem,
        "name": f"{num_nodes}_p_{sparsity:.2f}_{snap_file.stem}",
        "type": "snap",
        "graph": graph,
        "feature_matrix": sp.identity(num_nodes, format="csr", dtype=np.float32),
        "num_nodes": num_nodes,
        "sparsity": sparsity,
    }


def load_graph(graph_file):
    if "synthetic" in graph_file.parts:
        return load_synthetic(graph_file)
    elif "ssmatrices" in graph_file.parts:
        return load_matrix(graph_file)
    elif "snap" in graph_file.parts:
        return load_snap(graph_file)
    else:
        raise ValueError("Unknown graph type.")


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Run GNN experiments with CuPy sparse matrices")
    parser.add_argument("--method", "-m", type=str, help="Path of the method to run", required=True)
    parser.add_argument("--verify", default=False, action="store_true", help="Verify the result")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs")
    parser.add_argument(
        "--synthetic", "-sp", type=str, default=None, help="Index pattern of synthetic graphs to process"
    )
    parser.add_argument(
        "--ssmatrices", "-mp", type=str, default=None, help="Name pattern of SuiteSparse matrices to process"
    )
    parser.add_argument("--snap", "-np", type=str, default=None, help="Name pattern of SNAP networks to process")
    args = parser.parse_args()

    if not args.synthetic and not args.ssmatrices and not args.snap:
        parser.error("At least one of the following arguments must be provided: --synthetic, --ssmatrices, --snap")

    # import chosen method
    method = module_from_file(Path(args.method))

    # Load graphs or matrices
    graphs = [
        *parse_synthetic_pattern(args.synthetic),
        *parse_matrices_pattern(args.ssmatrices),
        *parse_snap_pattern(args.snap),
    ]

    results = defaultdict(dict)
    for graph_file in graphs:
        item = load_graph(graph_file)

        item_idx = str(item["index"])
        print(f"Testing {item_idx}...")

        # Execute the method
        result = method.execute(item, num_warmup=args.warmup)
        if result is None:
            print(f"Execution failed for {item_idx}.")
            continue

        # Verify the result
        is_correct = None
        if args.verify:
            adjacency_matrix = nx.to_scipy_sparse_array(item["graph"], format="lil", dtype=float)
            is_correct = bool(verify_result(result, adjacency_matrix, feature_matrix=item["feature_matrix"]))
            print(f"Verification: {'Correct' if is_correct else 'Incorrect'}")

        results[method.__name__][item_idx] = {
            "graph_name": item["name"],
            "graph_type": item["type"],
            "method": method.__name__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": item["num_nodes"],
            "sparsity": item["sparsity"],
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
