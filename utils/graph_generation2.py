import pickle
import shutil
from pathlib import Path

import networkx as nx
import numpy as np


# Function to clear the graphs directory
def clear_directory(directory):
    dir_path = Path(directory)
    if dir_path.exists():
        shutil.rmtree(dir_path)  # Deletes the directory and its contents
    dir_path.mkdir()  # Creates a new empty directory
    print(f"Cleared and created directory: {directory}")


# Generate synthetic graphs (Erdős-Rényi and Scale-Free)
def generate_synthetic_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type="both"):
    """
    Generate a list of synthetic graphs with varying sizes and sparsity levels.

    Parameters:
    num_graphs (int): Number of graphs per combination of size and sparsity.
    num_nodes_list (list): List of graph sizes (number of nodes).
    sparsity_levels (list): List of sparsity levels (between 0 and 1).

    Returns:
    list: List of NetworkX graph objects.
    """
    graphs = []
    print("Starting synthetic graph generation...")

    if gen_type == "both" or gen_type == "erdos-renyi":
        print("Generating Erdős-Rényi graphs...")
        for num_nodes in num_nodes_list:
            for p in sparsity_levels:
                for i in range(num_graphs):
                    G_er = nx.gnp_random_graph(num_nodes, p, directed=False)
                    node_mapping = {node: idx for idx, node in enumerate(G_er.nodes())}
                    G_er = nx.relabel_nodes(G_er, node_mapping)
                    graphs.append(
                        {
                            "graph": G_er,
                            "name": f"Erdos-Renyi_{num_nodes}_p_{p}_graph_{i+1}",
                            "type": "synthetic",
                            "num_nodes": G_er.number_of_nodes(),
                            "sparsity": p,
                        }
                    )
        print("Erdős-Rényi graphs generated successfully!")

    if gen_type == "both" or gen_type == "scale-free":
        print("Generating Scale-Free graphs...")
        for num_nodes in num_nodes_list:
            for i in range(num_graphs):
                G_sf = nx.barabasi_albert_graph(num_nodes, m=5)
                node_mapping = {node: idx for idx, node in enumerate(G_sf.nodes())}
                G_sf = nx.relabel_nodes(G_sf, node_mapping)
                graphs.append(
                    {
                        "graph": G_sf,
                        "name": f"Scale-Free_{num_nodes}_graph_{i+1}",
                        "type": "synthetic",
                        "num_nodes": G_sf.number_of_nodes(),
                        "sparsity": None,
                    }
                )
        print("Scale-Free graphs generated successfully!")

    return graphs


# Generate feature matrices for each graph
def generate_feature_matrices(graphs, num_features=10):
    """
    Generate a feature matrix for each graph and add it to the graph metadata.

    Parameters:
    graphs (list): List of graph metadata dictionaries.
    num_features (int): Number of features per node.

    Returns:
    list: List of graphs with feature matrices added.
    """
    print("Generating feature matrices for each graph...")
    for graph_data in graphs:
        num_nodes = graph_data["graph"].number_of_nodes()
        feature_matrix = np.random.rand(num_nodes, num_features).astype(np.float32)
        graph_data["feature_matrix"] = feature_matrix
    print("Feature matrices generated successfully!")
    return graphs


# Save graphs to file
def save_graphs(graphs, output_dir: Path):
    """
    Save the generated graphs to the specified directory.

    Parameters:
    graphs (list): List of graphs to save.
    """
    print(f"Saving graphs to '{output_dir}' directory...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, graph in enumerate(graphs):
        with open(output_dir / f"graph_{index}.pkl", "wb") as f:
            pickle.dump(
                {
                    "index": index,
                    "name": graph["name"],
                    "type": graph["type"],
                    "graph": graph["graph"],
                    "feature_matrix": graph["feature_matrix"],
                    "num_nodes": graph["num_nodes"],
                    "sparsity": graph["sparsity"],
                },
                f,
            )
    print(f"Saved {len(graphs)} graphs successfully!")


# Main execution block
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for graph generation
    num_graphs = 1
    num_nodes_list = [1000, 2000, 3000]
    sparsity_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
    number_of_features = 10
    g_type = "both"
    output_directory = Path("graphs")

    # Clear and create output directory
    clear_directory(output_directory)

    # Generate all graphs
    print("Generating synthetic graphs...")
    graphs = generate_synthetic_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type=g_type)

    # Generate feature matrices
    print("Generating feature matrices for each graph...")
    graphs = generate_feature_matrices(graphs, number_of_features)

    # Save the graphs
    save_graphs(graphs, output_directory)

    # Print final confirmation
    print(f"Generated and saved {len(graphs)} graphs with metadata and feature matrices.")
