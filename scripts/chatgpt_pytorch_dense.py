import nvtx
import torch
import numpy as np

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def execute(graph_info, num_warmup=1, num_runs=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]
    num_nodes = graph_info["num_nodes"]
    try:
        with nvtx.annotate(f"prepare {index}", domain="chatgpt_pytorch_dense"):
            feature_matrix = torch.tensor(feature_matrix.toarray(), dtype=torch.float32).to(device)

            # Prepare adjacency matrix
            num_nodes = feature_matrix.shape[0]
            adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32).to(device)
            for node in graph.nodes:
                for neighbor in graph.neighbors(node):
                    adjacency_matrix[node, neighbor] = 1.0

        # Warmup
        with nvtx.annotate(f"warmup {index}", domain="chatgpt_pytorch_dense"):
            for _ in range(num_warmup):
                result = torch.matmul(adjacency_matrix, feature_matrix)
                torch.cuda.synchronize()

        # Actual runs
        with nvtx.annotate(f"main {index}", domain="chatgpt_pytorch_dense"):
            times = []
            for _ in range(num_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                result = torch.matmul(adjacency_matrix, feature_matrix)
                end_event.record()
                torch.cuda.synchronize()
                
                times.append(start_event.elapsed_time(end_event))

        return result, np.mean(times), np.std(times)
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        # Free GPU memory
        torch.cuda.empty_cache()