import numpy as np
import scipy.sparse as sp
import pickle
import json
import time
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Load graphs
with open('gnn_test_graphs_with_features.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Define the PyG-based GCN layer and model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x

# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info['index']
    name = graph_info['name']
    type = graph_info['type']
    graph = graph_info['graph']
    num_nodes = graph_info['num_nodes']
    sparsity = graph_info['sparsity']

    feature_matrix = torch.tensor(graph_info['feature_matrix'].toarray(), dtype=torch.float32)
    print(f"Testing graph {index}")

    # Create a mapping from node labels to integers
    node_mapping = {node: i for i, node in enumerate(graph.nodes)}
    
    # Prepare PyG Data object
    edge_index = []
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            edge_index.append([node_mapping[node], node_mapping[neighbor]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=feature_matrix, edge_index=edge_index)
    
    # Initialize model
    model = GCN(in_channels=10, out_channels=10)
    
    # Perform forward pass and measure time
    start_time = time.time()
    model(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    results.append({
        'graph_index': index,
        'graph_name': name,
        'graph_type': type,
        'method': 'pytorch_geometric_gcn',
        'time_seconds': elapsed_time,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_nodes': num_nodes,
        'sparsity': sparsity
    })

import os
# Load existing results or create a new one
if os.path.exists('gnn_results.json'):
    with open('gnn_results.json', 'r') as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []  # Initialize as an empty list if the file is empty or corrupted
else:
    all_results = []

# Update results by replacing existing ones by graph index and method
for result in results:
    # Check if the result already exists in the list
    if any(r['graph_index'] == result['graph_index'] and r['method'] == result['method'] for r in all_results):
        # If so, replace the existing result
        all_results = [r for r in all_results if not (r['graph_index'] == result['graph_index'] and r['method'] == result['method'])]
        all_results.append(result)
    else:
        all_results.append(result)

# Save results
with open('gnn_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)

# Print confirmation
print("Results have been saved to 'gnn_results.json'.")
