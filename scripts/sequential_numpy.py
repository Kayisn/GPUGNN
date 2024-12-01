import time

import numpy as np


def execute(graph_info, num_warmup=1, num_runs=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    print(f"Testing graph {index}")

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0

    start_time = time.time()
    result = adjacency_matrix @ feature_matrix
    end_time = time.time()

    elapsed_time = end_time - start_time

    return result, elapsed_time, elapsed_time
