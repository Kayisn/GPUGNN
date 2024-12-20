import networkx as nx
import nvtx
import torch

# Set the CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def execute(graph_info, num_warmup=1):
    index = graph_info["index"]
    graph = graph_info["graph"]
    feature_matrix = graph_info["feature_matrix"]

    try:
        with nvtx.annotate(f"prepare {index}", domain="chatgpt_pytorch_dense"):
            feature_matrix = torch.tensor(feature_matrix.toarray(), dtype=torch.float32).to(device)

            # Prepare adjacency matrix
            adjacency_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32).to(device)

        # Warmup
        with nvtx.annotate(f"warmup {index}", domain="chatgpt_pytorch_dense"):
            for _ in range(num_warmup):
                result = torch.matmul(adjacency_matrix, feature_matrix)
                torch.cuda.synchronize()

        # Main
        with nvtx.annotate(f"main {index}", domain="chatgpt_pytorch_dense"):
            result = torch.matmul(adjacency_matrix, feature_matrix)
            torch.cuda.synchronize()

        return result.cpu().numpy()
    except Exception as e:
        print(f"Error processing graph: {e}")
    finally:
        torch.cuda.empty_cache()
