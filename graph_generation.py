import numpy as np
import networkx as nx
import pickle
import requests
import json
import scipy.sparse as sp

# Generate synthetic graphs (Erdős-Rényi and Scale-Free)
def generate_synthetic_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type='both'):
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
    if gen_type == 'both' or gen_type == 'erdos-renyi':
        # Generate Erdős-Rényi graphs
        for num_nodes in num_nodes_list:
            for p in sparsity_levels:
                for i in range(num_graphs):
                    G_er = nx.gnp_random_graph(num_nodes, p, directed=False)
                    node_mapping = {node: idx for idx, node in enumerate(G_er.nodes())}
                    G_er = nx.relabel_nodes(G_er, node_mapping)
                    graphs.append({'graph': G_er, 'name': f'Erdos-Renyi_{num_nodes}_p_{p}_graph_{i+1}', 'type': 'synthetic', 'num_nodes': G_er.number_of_nodes(), 'sparsity': p})
    if gen_type == 'both' or gen_type == 'scale-free':    
        # Generate Scale-Free graphs
        for num_nodes in num_nodes_list:
            for i in range(num_graphs):
                G_sf = nx.barabasi_albert_graph(num_nodes, m=5)
                node_mapping = {node: idx for idx, node in enumerate(G_sf.nodes())}
                G_sf = nx.relabel_nodes(G_sf, node_mapping)
                graphs.append({'graph': G_sf, 'name': f'Scale-Free_{num_nodes}_graph_{i+1}', 'type': 'synthetic', 'num_nodes': G_sf.number_of_nodes(), 'sparsity': None})
    return graphs

# Function to add other real-world graphs available in NetworkX
def add_real_world_graphs(num_samples=3, sample_size=200):
    """
    Add real-world graphs from NetworkX library and take samples to create smaller subgraphs.

    Returns:
    list: List of NetworkX graph objects representing real-world graphs and their sampled subgraphs.
    """
    graphs = []

    # Example: Karate Club social network
    G_karate = nx.karate_club_graph()
    node_mapping = {node: idx for idx, node in enumerate(G_karate.nodes())}
    G_karate = nx.relabel_nodes(G_karate, node_mapping)
    graphs.append({'graph': G_karate, 'name': 'Karate-Club', 'type': 'real-world', 'num_nodes': G_karate.number_of_nodes(), 'sparsity': None})
    # Generate samples from the Karate Club graph
    for i in range(num_samples):
        sampled_nodes = np.random.choice(list(G_karate.nodes()), size=min(sample_size, G_karate.number_of_nodes()), replace=False)
        G_sample = G_karate.subgraph(sampled_nodes).copy()
        graphs.append({'graph': G_sample, 'name': f'Karate-Club-Sample-{i+1}', 'type': 'real-world-sample', 'num_nodes': G_sample.number_of_nodes(), 'sparsity': None})
    print("Successfully added Karate Club social network.")

    # Example: Les Miserables character network
    G_les_miserables = nx.les_miserables_graph()
    node_mapping = {node: idx for idx, node in enumerate(G_les_miserables.nodes())}
    G_les_miserables = nx.relabel_nodes(G_les_miserables, node_mapping)
    graphs.append({'graph': G_les_miserables, 'name': 'Les-Miserables', 'type': 'real-world', 'num_nodes': G_les_miserables.number_of_nodes(), 'sparsity': None})
    # Generate samples from the Les Miserables graph
    for i in range(num_samples):
        sampled_nodes = np.random.choice(list(G_les_miserables.nodes()), size=min(sample_size, G_les_miserables.number_of_nodes()), replace=False)
        G_sample = G_les_miserables.subgraph(sampled_nodes).copy()
        graphs.append({'graph': G_sample, 'name': f'Les-Miserables-Sample-{i+1}', 'type': 'real-world-sample', 'num_nodes': G_sample.number_of_nodes(), 'sparsity': None})
    print("Successfully added Les Miserables character network.")

    return graphs

# Generate and collect all graphs
def generate_all_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type='both'):
    """
    Generate and download graphs for testing.
    
    Parameters:
    num_graphs (int): Number of graphs per combination of size and sparsity.
    num_nodes_list (list): List of graph sizes (number of nodes).
    sparsity_levels (list): List of sparsity levels for Erdős-Rényi graphs.

    Returns:
    list: List of NetworkX graph objects.
    """
    synthetic_graphs = generate_synthetic_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type=gen_type)
    #real_world_graphs = add_real_world_graphs(num_samples=3, sample_size=1000)
    #return synthetic_graphs + real_world_graphs
    return synthetic_graphs

def create_solution_matrix(graph, feature_matrix):
    """
    Create a solution matrix for a graph by performing a brute-force matrix multiplication.

    Parameters:
    graph (nx.Graph): The graph for which to create the solution matrix.
    feature_matrix (sp.csr_matrix): The feature matrix for the graph.

    Returns:
    sp.csr_matrix: The solution matrix for the graph.
    """
    num_nodes = graph.number_of_nodes()
    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0
    return adjacency_matrix.dot(feature_matrix)


# Generate feature matrices for each graph
def generate_feature_matrices(graphs, num_features=10):
    """
    Generate a feature matrix for each graph and add it to the graph metadata.

    Parameters:
    graphs (list): List of graph metadata dictionaries.
    num_features (int): Number of features per node.

    Returns:
    list: List of graphs with feature matrices and brute force multiplication results added.
    """
    for graph_data in graphs:
        num_nodes = graph_data['graph'].number_of_nodes()
        feature_matrix = sp.csr_matrix(np.random.rand(num_nodes, num_features).astype(np.float32))
        graph_data['feature_matrix'] = feature_matrix
    return graphs


# Parameters for graph generation

filename = 'gnn_test_graphs_with_features.pkl'
num_graphs = 1
num_nodes_list = [1000, 2000, 3000]
sparsity_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
number_of_features = 10
g_type = 'both'
"""
filename = 'gnn_test_graph_with_features.pkl'
num_graphs = 1
num_nodes_list = [100000]
sparsity_levels = [0.05]
number_of_features = 10
g_type = 'scale-free'
"""

# Generate all graphs
graphs = generate_all_graphs(num_graphs, num_nodes_list, sparsity_levels, gen_type=g_type)

# Generate feature matrices for each graph
graphs = generate_feature_matrices(graphs, number_of_features)

# Save graphs to file
def save_graphs(graphs, filename):
    """
    Save the generated graphs to a file using pickle.
    
    Parameters:
    graphs (list): List of graphs to save.
    filename (str): The filename to save the graphs to.
    """
    with open(filename, 'wb') as f:
        pickle.dump([{'index': index, 'name': g['name'], 'type': g['type'], 'graph': g['graph'], 'feature_matrix': g['feature_matrix'], 'num_nodes': g['num_nodes'], 'sparsity': g['sparsity']} for index, g in enumerate(graphs)], f)

# Save the graphs
save_graphs(graphs, filename)

# Print confirmation
print(f"Generated and saved {len(graphs)} graphs including synthetic, real-world, and sampled subgraphs with metadata and feature matrices.")