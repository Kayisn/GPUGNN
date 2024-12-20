import numpy as np
import scipy.sparse as sp
import pickle

import networkx as nx

def generate_erdos_renyi_graph(num_nodes, p):
    """
    Generate a large Erdős-Rényi graph using sparse matrices.

    Parameters:
    num_nodes (int): Number of nodes in the graph.
    p (float): Probability of edge creation.

    Returns:
    sp.csr_matrix: Adjacency matrix of the graph.
    """
    rng = np.random.default_rng()
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = int(p * num_possible_edges)
    row_indices = rng.integers(0, num_nodes, size=num_edges)
    col_indices = rng.integers(0, num_nodes, size=num_edges)
    # Remove self-loops
    mask = row_indices != col_indices
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    data = np.ones(len(row_indices), dtype=np.float32)
    adjacency = sp.coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    adjacency = sp.triu(adjacency, k=1)
    adjacency = adjacency + adjacency.T
    return adjacency.tocsr()

def generate_scale_free_graph(num_nodes, m):
    """
    Generate a large Scale-Free graph using the Barabási-Albert model with optimizations.

    Parameters:
    num_nodes (int): Number of nodes in the graph.
    m (int): Number of edges to attach from a new node to existing nodes.

    Returns:
    sp.csr_matrix: Adjacency matrix of the graph.
    """
    # Initial connected network of m nodes
    sources = np.zeros(m * num_nodes, dtype=np.int32)
    targets = np.zeros(m * num_nodes, dtype=np.int32)
    degrees = np.zeros(num_nodes, dtype=np.int32)
    degree_sum = 0

    # Create a complete graph with m nodes
    for i in range(m):
        for j in range(i):
            idx = i * m + j
            sources[idx] = i
            targets[idx] = j
            degrees[i] += 1
            degrees[j] += 1
            degree_sum += 2  # Undirected graph

    # Starting index for sources and targets
    edge_idx = m * (m - 1) // 2

    # Precompute cumulative sum of degrees for efficient sampling
    for source in range(m, num_nodes):
        probs = degrees[:source] / degree_sum
        targets_idx = np.random.choice(source, size=m, replace=False, p=probs)
        idx_slice = slice(edge_idx, edge_idx + m)
        sources[idx_slice] = source
        targets[idx_slice] = targets_idx
        degrees[source] += m
        degrees[targets_idx] += 1
        degree_sum += 2 * m
        edge_idx += m

    sources = sources[:edge_idx]
    targets = targets[:edge_idx]

    # Create adjacency matrix
    data = np.ones(len(sources), dtype=np.float32)
    adjacency = sp.coo_matrix((data, (sources, targets)), shape=(num_nodes, num_nodes))
    adjacency = adjacency + adjacency.T
    return adjacency.tocsr()

def generate_feature_matrix(num_nodes, num_features):
    """
    Generate a feature matrix for the nodes.

    Parameters:
    num_nodes (int): Number of nodes.
    num_features (int): Number of features per node.

    Returns:
    sp.csr_matrix: Feature matrix.
    """
    rng = np.random.default_rng()
    features = rng.random((num_nodes, num_features), dtype=np.float32)
    return sp.csr_matrix(features)

# Parameters for graph generation
filename = 'gnn_test_graphs_with_features.pkl'
num_graphs = 2
num_nodes_list = [100000]
sparsity_levels = [0.0001, 0.001]
number_of_features = 10
graph_type = 'erdos-renyi'  # 'erdos-renyi' or 'scale-free'

graphs = []
for num_nodes in num_nodes_list:
    for i in range(num_graphs):
        if graph_type == 'erdos-renyi':
            for p in sparsity_levels:
                adjacency = generate_erdos_renyi_graph(num_nodes, p)
                feature_matrix = generate_feature_matrix(num_nodes, number_of_features)
                graph = nx.from_scipy_sparse_array(adjacency)
                name = f'Erdos-Renyi_{num_nodes}_p_{p}_graph_{i+1}'
                graphs.append({
                    'index': len(graphs),
                    'name': name,
                    'type': 'synthetic',
                    'adjacency': adjacency,
                    'graph': graph,
                    'feature_matrix': feature_matrix,
                    'num_nodes': num_nodes,
                    'sparsity': p
                })
                print(f"Generated {name}")
        elif graph_type == 'scale-free':
            m = 5  # Number of edges to attach from a new node
            adjacency = generate_scale_free_graph(num_nodes, m)
            feature_matrix = generate_feature_matrix(num_nodes, number_of_features)
            graph = nx.from_scipy_sparse_array(adjacency)
            name = f'Scale-Free_{num_nodes}_graph_{i+1}'
            graphs.append({
                'index': len(graphs),
                'name': name,
                'type': 'synthetic',
                'adjacency': adjacency,
                'graph': graph,
                'feature_matrix': feature_matrix,
                'num_nodes': num_nodes,
                'sparsity': None
            })
            print(f"Generated {name}")



# Save all graphs to a single file
with open(filename, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Generated and saved {len(graphs)} graphs to {filename}")