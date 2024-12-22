import argparse
import csv
import gzip
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


def parse_graph_indices(graphs_pattern):
    if graphs_pattern == "all":
        return synthetic_dir.glob("graph_*.pkl")
    if "," in graphs_pattern:
        return [synthetic_dir / f"graph_{i}.pkl" for i in graphs_pattern.split(",")]
    if "-" in graphs_pattern:
        start, end = map(int, graphs_pattern.split("-"))
        return [synthetic_dir / f"graph_{i}.pkl" for i in range(start, end + 1)]
    return [synthetic_dir / f"graph_{graphs_pattern}.pkl"]


def parse_matrices_indices(matrices_pattern):
    matrices = ssmatrices_dir.glob("*.mtx")
    if matrices_pattern != "all":
        for pattern in matrices_pattern.split(","):
            matrices = filter(lambda m: pattern in m.stem, matrices)
    return matrices


def parse_snap_indices(matrices_pattern):
    matrices = snap_dir.glob("*.gz")
    if matrices_pattern != "all":
        for pattern in matrices_pattern.split(","):
            matrices = filter(lambda m: pattern in m.stem, matrices)
    return matrices


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
    data = []

    if args.synthetic:
        print(f"Loading graphs...")
        for graph_file in parse_graph_indices(args.synthetic):
            with open(graph_file, "rb") as f:
                data.append(pickle.load(f))

    if args.ssmatrices:
        print(f"Loading matrices...")
        for matrix_file in parse_matrices_indices(args.ssmatrices):
            matrix = scipy.io.mmread(matrix_file).astype(np.float32)
            num_nodes = matrix.shape[0]
            sparsity = 1 - (matrix.nnz / (num_nodes**2))
            data.append(
                {
                    "index": matrix_file.stem,
                    "name": f"{num_nodes}_p_{sparsity:.2f}_{matrix_file.stem}",
                    "type": "matrix",
                    "graph": nx.from_scipy_sparse_array(matrix),
                    "feature_matrix": matrix,
                    "num_nodes": num_nodes,
                    "sparsity": sparsity,
                }
            )

    if args.snap:
        print(f"Loading SNAP networks...")
        for snap_file in parse_snap_indices(args.snap):
            if ".txt" in snap_file.suffixes:
                graph = nx.read_edgelist(snap_file, nodetype=int)
            elif ".csv" in snap_file.suffixes:
                graph = nx.read_edgelist(
                    snap_file, delimiter=",", nodetype=int, data=(("source", int), ("target", int))
                )

            num_nodes = graph.number_of_nodes()
            sparsity = 1 - (graph.number_of_edges() / (num_nodes**2))
            data.append(
                {
                    "index": snap_file.stem,
                    "name": f"{num_nodes}_p_{sparsity:.2f}_{snap_file.stem}",
                    "type": "snap",
                    "graph": graph,
                    "feature_matrix": sp.identity(num_nodes, format="csr", dtype=np.float32),
                    "num_nodes": num_nodes,
                    "sparsity": sparsity,
                }
            )

    results = defaultdict(dict)
    for item in data:
        item_idx = str(item["index"])
        print(f"Testing {item_idx}...")

        # Execute the method
        result = method.execute(item, num_warmup=args.warmup)

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
