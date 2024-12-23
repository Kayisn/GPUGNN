"""
CLAUDE 3.5
The key advantages of this implementation are:

Better memory coalescing for edge access
Reduced thread divergence
More efficient parallel processing of edges
Better GPU utilization for sparse graphs
Simplified memory management
Lower overhead compared to clustering approaches
"""

import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty  # Import Empty explicitly

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule
import numpy as np
from numpy import random as np_random  # Add this import

import concurrent
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
import random
from deap import base, creator, tools, algorithms

from verification import verify_result

import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm, skew, kurtosis

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

import os

# Set CUDA compiler path before importing pycuda
os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']

# Memory tracking thread function
def memory_monitor(stop_event, context):
    peak_memory_usage = 0
    context.push()  # Push the context to the current thread
    while not stop_event.is_set():
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = total_mem - free_mem
        peak_memory_usage = max(peak_memory_usage, used_mem)
        time.sleep(0.1)  # Sleep for a short duration to avoid busy-waiting
    context.pop()  # Pop the context from the current thread
    return peak_memory_usage

def get_gpu_capabilities():
    """Check GPU capabilities including tensor core support"""
    device = cuda.Device(0)
    attributes = device.get_attributes()
    
    # Check for Tensor Core support (SM 7.0 or higher)
    compute_capability = device.compute_capability()
    has_tensor_cores = compute_capability[0] >= 7

    return {
        'has_tensor_cores': has_tensor_cores,
        'compute_capability': compute_capability,
        'total_memory': device.total_memory()
    }



# New CUDA kernel optimized for edge processing
EDGE_THROUGHPUT_KERNEL = """
__global__ void edge_based_matmul(
    const float* __restrict__ edge_values,
    const int* __restrict__ edge_sources,
    const int* __restrict__ edge_targets,
    const float* __restrict__ feature_matrix,
    float* __restrict__ output_matrix,
    const int batch_start,
    const int batch_size,
    const int feature_dim,
    const int num_nodes
) {
    // Process edges in parallel with coalesced memory access
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    int edge_idx = batch_start + tid;
    int src = edge_sources[edge_idx];
    int dst = edge_targets[edge_idx];
    float edge_val = edge_values[edge_idx];
    
    // Each thread processes multiple features for better GPU utilization
    for (int feat = threadIdx.y; feat < feature_dim; feat += blockDim.y) {
        atomicAdd(&output_matrix[dst * feature_dim + feat],
                 edge_val * feature_matrix[src * feature_dim + feat]);
    }
}
"""

def prepare_edge_lists(adjacency_matrix):
    """Convert adjacency matrix to edge lists for efficient GPU processing"""
    rows, cols = adjacency_matrix.nonzero()
    edge_values = adjacency_matrix.data.astype(np.float32)
    edge_sources = rows.astype(np.int32)
    edge_targets = cols.astype(np.int32)
    return edge_values, edge_sources, edge_targets

def sparse_matrix_multiply_edge_centric(adjacency_matrix, feature_matrix, batch_size=1024, feature_matrix_gpu=None):
    """Edge-centric sparse matrix multiplication with optional pre-allocated feature matrix"""
    num_nodes = adjacency_matrix.shape[0]
    num_features = feature_matrix.shape[1]
    
    # Convert to edge representation
    edge_values, edge_sources, edge_targets = prepare_edge_lists(adjacency_matrix)
    num_edges = len(edge_values)
    
    # Compile kernel
    mod = SourceModule(EDGE_THROUGHPUT_KERNEL)
    edge_matmul = mod.get_function("edge_based_matmul")
    
    # Allocate GPU memory
    edge_values_gpu = cuda.mem_alloc(edge_values.nbytes)
    edge_sources_gpu = cuda.mem_alloc(edge_sources.nbytes)
    edge_targets_gpu = cuda.mem_alloc(edge_targets.nbytes)
    output_matrix_gpu = cuda.mem_alloc(num_nodes * num_features * 4)
    
    # Copy data to GPU
    cuda.memcpy_htod(edge_values_gpu, edge_values)
    cuda.memcpy_htod(edge_sources_gpu, edge_sources)
    cuda.memcpy_htod(edge_targets_gpu, edge_targets)
    
    # Use provided feature matrix GPU memory or allocate new
    should_free_feature_matrix = False
    if feature_matrix_gpu is None:
        feature_matrix = feature_matrix.astype(np.float32)
        feature_matrix_gpu = cuda.mem_alloc(feature_matrix.nbytes)
        cuda.memcpy_htod(feature_matrix_gpu, feature_matrix)
        should_free_feature_matrix = True
    
    # Initialize output matrix to zeros
    cuda.memset_d32(output_matrix_gpu, 0, num_nodes * num_features)
    
    try:
        # Process edges in batches
        for batch_start in range(0, num_edges, batch_size):
            batch_end = min(batch_start + batch_size, num_edges)
            current_batch_size = batch_end - batch_start
            
            threads_per_block = (256, 4, 1)
            num_blocks = ((current_batch_size + threads_per_block[0] - 1) // threads_per_block[0], 1, 1)
            
            edge_matmul(
                edge_values_gpu,
                edge_sources_gpu,
                edge_targets_gpu,
                feature_matrix_gpu,
                output_matrix_gpu,
                np.int32(batch_start),
                np.int32(current_batch_size),
                np.int32(num_features),
                np.int32(num_nodes),
                block=threads_per_block,
                grid=num_blocks
            )
        
        # Copy result back to host
        result = np.empty((num_nodes, num_features), dtype=np.float32)
        cuda.memcpy_dtoh(result, output_matrix_gpu)
        return result
        
    finally:
        # Clean up
        edge_values_gpu.free()
        edge_sources_gpu.free()
        edge_targets_gpu.free()
        output_matrix_gpu.free()
        if should_free_feature_matrix:
            feature_matrix_gpu.free()

def bfs_edge_decomposition(edge_values, edge_sources, edge_targets, max_edges_per_batch):
    """BFS-based edge decomposition strategy"""
    G = nx.Graph()
    for i in range(len(edge_sources)):
        G.add_edge(edge_sources[i], edge_targets[i], weight=edge_values[i])
    
    batches = []
    visited_edges = set()
    
    for start_node in G.nodes():
        if len(visited_edges) >= len(edge_sources):
            break
            
        edge_batch = ([], [], [])
        edge_queue = []
        
        # BFS traversal
        bfs_edges = list(nx.bfs_edges(G, start_node))
        for u, v in bfs_edges:
            if (u, v) not in visited_edges and len(edge_batch[0]) < max_edges_per_batch:
                edge_data = G.get_edge_data(u, v)
                edge_batch[0].append(edge_data['weight'])
                edge_batch[1].append(u)
                edge_batch[2].append(v)
                visited_edges.add((u, v))
                visited_edges.add((v, u))
        
        if edge_batch[0]:
            batches.append((
                np.array(edge_batch[0], dtype=np.float32),
                np.array(edge_batch[1], dtype=np.int32),
                np.array(edge_batch[2], dtype=np.int32)
            ))
    
    return batches

def spectral_edge_decomposition(edge_values, edge_sources, edge_targets, num_clusters):
    """Spectral clustering based edge decomposition"""
    # Create adjacency matrix for spectral clustering
    num_nodes = max(max(edge_sources), max(edge_targets)) + 1
    adj_matrix = sp.csr_matrix(
        (edge_values, (edge_sources, edge_targets)), 
        shape=(num_nodes, num_nodes)
    )
    
    # Perform spectral clustering
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=42
    )
    node_clusters = clustering.fit_predict(adj_matrix.toarray())
    
    # Group edges by cluster
    batches = [[] for _ in range(num_clusters)]
    for i in range(len(edge_sources)):
        src_cluster = node_clusters[edge_sources[i]]
        batches[src_cluster].append((edge_values[i], edge_sources[i], edge_targets[i]))
    
    # Convert to numpy arrays
    return [
        (
            np.array([e[0] for e in batch], dtype=np.float32),
            np.array([e[1] for e in batch], dtype=np.int32),
            np.array([e[2] for e in batch], dtype=np.int32)
        )
        for batch in batches if batch
    ]

def locality_aware_decomposition(edge_values, edge_sources, edge_targets, max_edges_per_batch):
    """Locality-sensitive decomposition using node proximity"""
    edges = list(zip(edge_values, edge_sources, edge_targets))
    edges.sort(key=lambda x: x[1])  # Sort by source node
    
    batches = []
    current_batch = []
    
    for edge in edges:
        current_batch.append(edge)
        if len(current_batch) >= max_edges_per_batch:
            batches.append(
                (
                    np.array([e[0] for e in current_batch], dtype=np.float32),
                    np.array([e[1] for e in current_batch], dtype=np.int32),
                    np.array([e[2] for e in current_batch], dtype=np.int32)
                )
            )
            current_batch = []
    
    if current_batch:
        batches.append(
            (
                np.array([e[0] for e in current_batch], dtype=np.float32),
                np.array([e[1] for e in current_batch], dtype=np.int32),
                np.array([e[2] for e in current_batch], dtype=np.int32)
            )
        )
    
    return batches

# Evolutionary optimization components
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Add new GPU-based decomposition kernels
PAGERANK_KERNEL = """
extern "C" __global__ void pagerank_edge_cut(
    const float* __restrict__ node_scores,
    const int* __restrict__ edge_sources,
    const int* __restrict__ edge_targets,
    const float* __restrict__ edge_weights,
    float* __restrict__ edge_scores,
    const int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int src = edge_sources[tid];
        int dst = edge_targets[tid];
        edge_scores[tid] = edge_weights[tid] * (node_scores[src] + node_scores[dst]);
    }
}
"""

PARTITION_KERNEL = """
extern "C" __global__ void partition_scoring(
    const float* __restrict__ edge_weights,
    const int* __restrict__ edge_sources,
    const int* __restrict__ edge_targets,
    const int* __restrict__ node_partitions,
    float* __restrict__ partition_scores,
    const int num_edges,
    const int num_partitions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int src_part = node_partitions[edge_sources[tid]];
        int dst_part = node_partitions[edge_targets[tid]];
        if (src_part != dst_part) {
            atomicAdd(&partition_scores[src_part * num_partitions + dst_part], 
                     edge_weights[tid]);
        }
    }
}
"""

def pagerank_edge_decomposition(edge_values, edge_sources, edge_targets, num_nodes, max_edges_per_batch):
    """GPU-accelerated PageRank-based edge decomposition with edge cuts"""
    # Create adjacency matrix for PageRank
    adj_matrix = sp.csr_matrix(
        (edge_values, (edge_sources, edge_targets)),
        shape=(num_nodes, num_nodes)
    )
    
    # Normalize adjacency matrix for PageRank
    row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    row_normalize = sp.diags(1.0 / row_sums)
    adj_matrix = row_normalize @ adj_matrix
    
    # Initialize PageRank on CPU
    d = 0.85
    max_iter = 100
    tol = 1e-6
    
    scores = np.ones(num_nodes, dtype=np.float32) / num_nodes
    prev_scores = scores.copy()
    
    # PageRank iteration with better numerical stability
    for _ in range(max_iter):
        scores_new = (1 - d) / num_nodes + d * (adj_matrix @ scores)
        diff = np.abs(scores_new - prev_scores).max()
        if diff < tol:
            break
        prev_scores = scores.copy()
        scores = scores_new
    
    # Allocate GPU memory for edge scoring
    scores_gpu = cuda.mem_alloc(scores.nbytes)
    edge_scores_gpu = cuda.mem_alloc(len(edge_values) * 4)  # float32
    cuda.memcpy_htod(scores_gpu, scores)
    
    # Compile edge scoring kernel
    mod = SourceModule(PAGERANK_KERNEL)
    edge_scorer = mod.get_function('pagerank_edge_cut')
    
    # Score edges based on PageRank values
    edge_scorer(
        scores_gpu,
        cuda.In(edge_sources),
        cuda.In(edge_targets),
        cuda.In(edge_values),
        edge_scores_gpu,
        np.int32(len(edge_values)),
        block=(256, 1, 1),
        grid=(min(65535, (len(edge_values) + 255) // 256), 1)
    )
    
    # Copy scores back and sort
    edge_scores = np.empty(len(edge_values), dtype=np.float32)
    cuda.memcpy_dtoh(edge_scores, edge_scores_gpu)
    sorted_indices = np.argsort(edge_scores)
    
    # Create batches
    batches = []
    current_batch = ([], [], [])
    
    for idx in sorted_indices:
        current_batch[0].append(edge_values[idx])
        current_batch[1].append(edge_sources[idx])
        current_batch[2].append(edge_targets[idx])
        
        if len(current_batch[0]) >= max_edges_per_batch:
            batches.append((
                np.array(current_batch[0], dtype=np.float32),
                np.array(current_batch[1], dtype=np.int32),
                np.array(current_batch[2], dtype=np.int32)
            ))
            current_batch = ([], [], [])
    
    if current_batch[0]:
        batches.append((
            np.array(current_batch[0], dtype=np.float32),
            np.array(current_batch[1], dtype=np.int32),
            np.array(current_batch[2], dtype=np.int32)
        ))
    
    # Cleanup
    scores_gpu.free()
    edge_scores_gpu.free()
    
    return batches

def metis_style_decomposition(edge_values, edge_sources, edge_targets, num_nodes, num_partitions):
    """GPU-accelerated METIS-style graph partitioning"""
    # Initialize random partitions
    node_partitions = np.random.randint(0, num_partitions, num_nodes, dtype=np.int32)
    
    # Allocate GPU memory
    node_partitions_gpu = cuda.mem_alloc(node_partitions.nbytes)
    partition_scores_gpu = cuda.mem_alloc(num_partitions * num_partitions * 4)  # float32
    
    # Compile partition scoring kernel
    mod = SourceModule(PARTITION_KERNEL)
    partition_scorer = mod.get_function('partition_scoring')
    
    # Iterative refinement
    max_iter = 20
    for _ in range(max_iter):
        cuda.memcpy_htod(node_partitions_gpu, node_partitions)
        cuda.memset_d32(partition_scores_gpu, 0, num_partitions * num_partitions)
        
        # Score partitions
        partition_scorer(
            cuda.In(edge_values),
            cuda.In(edge_sources),
            cuda.In(edge_targets),
            node_partitions_gpu,
            partition_scores_gpu,
            np.int32(len(edge_values)),
            np.int32(num_partitions),
            block=(256, 1, 1),
            grid=(min(65535, (len(edge_values) + 255) // 256), 1)
        )
        
        # Copy scores back
        partition_scores = np.empty((num_partitions, num_partitions), dtype=np.float32)
        cuda.memcpy_dtoh(partition_scores, partition_scores_gpu)
        
        # Update partitions based on scores
        node_counts = np.bincount(node_partitions, minlength=num_partitions)
        partition_costs = partition_scores.sum(axis=1) / np.maximum(node_counts, 1)
        
        # Move nodes to better partitions
        for node in range(num_nodes):
            current_part = node_partitions[node]
            best_part = current_part
            best_cost = partition_costs[current_part]
            
            for new_part in range(num_partitions):
                if new_part != current_part:
                    cost = partition_costs[new_part]
                    if cost < best_cost and node_counts[new_part] < 1.1 * (len(edge_sources) / num_partitions):
                        best_cost = cost
                        best_part = new_part
            
            if best_part != current_part:
                node_partitions[node] = best_part
                node_counts[current_part] -= 1
                node_counts[best_part] += 1
    
    # Create batches based on partitions
    batches = []
    for partition in range(num_partitions):
        mask = np.logical_or(
            node_partitions[edge_sources] == partition,
            node_partitions[edge_targets] == partition
        )
        
        if np.any(mask):
            batches.append((
                edge_values[mask],
                edge_sources[mask],
                edge_targets[mask]
            ))
    
    # Cleanup
    node_partitions_gpu.free()
    partition_scores_gpu.free()
    
    return batches

class GraphTopologyLearner:
    def __init__(self, model_path="topology_model.joblib"):
        self.model_path = model_path
        self.feature_names = [
            'num_nodes', 'num_edges', 'sparsity',
            'degree_mean', 'degree_std', 'degree_skew',
            'clustering_sample', 'eigenvector_sample',
            'edge_locality'
        ]
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            self.model = joblib.load(self.model_path)
            self.training_data = joblib.load(self.model_path + ".data")
        except:
            self.model = RandomForestRegressor(n_estimators=100)
            self.training_data = {
                'X': [],
                'y': [],
                'performance': []
            }
    
    def extract_topology_features(self, adjacency_matrix, edge_values, edge_sources, edge_targets):
        """Extract key features describing graph topology"""
        num_nodes = adjacency_matrix.shape[0]
        num_edges = len(edge_values)
        sparsity = num_edges / (num_nodes * num_nodes)
        
        # Degree statistics
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degree_mean = degrees.mean()
        degree_std = degrees.std()
        degree_skew = skew(degrees)
        
        # Sample-based clustering coefficient (for efficiency)
        sample_size = min(1000, num_nodes)
        sample_nodes = np.random.choice(num_nodes, sample_size, replace=False)
        clustering_sample = 0
        for node in sample_nodes:
            neighbors = adjacency_matrix[node].indices
            if len(neighbors) > 1:
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = adjacency_matrix[neighbors][:, neighbors].sum() / 2
                clustering_sample += actual_edges / possible_edges
        clustering_sample /= sample_size
        
        # Approximate eigenvector centrality
        num_iter = 10
        centrality = np.ones(num_nodes)
        for _ in range(num_iter):
            centrality_new = adjacency_matrix @ centrality
            centrality = centrality_new / np.linalg.norm(centrality_new)
        eigenvector_sample = centrality[sample_nodes].mean()
        
        # Edge locality measure
        source_diffs = np.diff(np.sort(edge_sources))
        edge_locality = (source_diffs == 1).mean()
        
        return np.array([
            num_nodes, num_edges, sparsity,
            degree_mean, degree_std, degree_skew,
            clustering_sample, eigenvector_sample,
            edge_locality
        ])
    
    def predict_initial_params(self, features):
        """Predict good initial parameters for evolutionary search"""
        if len(self.training_data['X']) < 10:
            # Not enough data, return default parameters
            return {
                'batch_size': 512,
                'num_clusters': 10,
                'strategy': 2  # Locality-aware as default
            }
        
        prediction = self.model.predict([features])[0]
        return {
            'batch_size': int(prediction[0] * 1024),
            'num_clusters': int(prediction[1] * 20),
            'strategy': int(prediction[2] * self.num_strategies)
        }
    
    def update_model(self, features, params, performance):
        """Update model with new training data"""
        self.training_data['X'].append(features)
        self.training_data['y'].append([
            params['batch_size'] / 1024,
            params['num_clusters'] / 20,
            params['strategy'] / self.num_strategies
        ])
        self.training_data['performance'].append(performance)
        
        # Retrain model if enough new data
        if len(self.training_data['X']) >= 10:
            X = np.array(self.training_data['X'])
            y = np.array(self.training_data['y'])
            self.model.fit(X, y)
            
            # Save updated model and data
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.training_data, self.model_path + ".data")

class DecompositionOptimizer:
    def __init__(self, edge_values, edge_sources, edge_targets, num_nodes, adjacency_matrix):
        self.edge_values = edge_values
        self.edge_sources = edge_sources
        self.edge_targets = edge_targets
        self.num_nodes = num_nodes
        self.adjacency_matrix = adjacency_matrix
        self.learner = GraphTopologyLearner()

        # Define evolution parameters
        self.pop_size = 4  # Reduced population size
        self.num_generations = 3  # Reduced generations
        self.num_islands = 2  # Reduced islands

        # Update number of strategies to 5
        self.num_strategies = 4
        
        # Get topology features and initial parameters
        self.topology_features = self.learner.extract_topology_features(
            adjacency_matrix, edge_values, edge_sources, edge_targets)
        initial_params = self.learner.predict_initial_params(self.topology_features)
        
        # Initialize population with learned parameters
        #self.init_population_with_params(initial_params)
        
        # Setup genetic algorithm toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0.1, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=3)
        self.toolbox.register("population", tools.initRepeat, list,
                            self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_decomposition)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    
    def init_population_with_params(self, params):
        """Initialize part of population with learned parameters"""
        # Convert parameters back to [0,1] range
        learned_individual = [
            params['batch_size'] / 1024,
            params['num_clusters'] / 20,
            params['strategy'] / self.num_strategies
        ]
        
        # Add some noise to create variations
        noise_scale = 0.1
        learned_pop = []
        for _ in range(self.pop_size // 4):  # 25% of population from learned params
            noisy_individual = np.clip(
                learned_individual + np.random.normal(0, noise_scale, 3),
                0.1, 1.0
            )
            learned_pop.append(creator.Individual(noisy_individual))
        
        # Rest of population random as before
        random_pop = [self.toolbox.individual() 
                     for _ in range(self.pop_size - len(learned_pop))]
        
        self.initial_population = learned_pop + random_pop
    
    def evaluate_decomposition(self, individual):
        """Evaluate decomposition strategy performance"""
        batch_size = int(individual[0] * 1024)  # Scale between 0-1024
        num_clusters = int(individual[1] * 20)  # Scale between 0-20
        strategy = int(individual[2] * self.num_strategies)  # Choose between 5 strategies
        
        try:
            if strategy == 0:
                batches = bfs_edge_decomposition(
                    self.edge_values, self.edge_sources, self.edge_targets, batch_size)
            elif strategy == 1:
                batches = spectral_edge_decomposition(
                    self.edge_values, self.edge_sources, self.edge_targets, 
                    max(2, num_clusters))
            elif strategy == 2:
                batches = locality_aware_decomposition(
                    self.edge_values, self.edge_sources, self.edge_targets, batch_size)
            elif strategy == -1:
                batches = pagerank_edge_decomposition(
                    self.edge_values, self.edge_sources, self.edge_targets,
                    self.num_nodes, batch_size)
            else:
                batches = metis_style_decomposition(
                    self.edge_values, self.edge_sources, self.edge_targets,
                    self.num_nodes, max(2, num_clusters))
            
            # Time the execution
            start_time = time.perf_counter()
            with CUDAContextManager.get_context():
                pipeline = EdgePipeline(batch_size=batch_size)
                _ = pipeline.process_edges(batches, feature_matrix, self.num_nodes)
            execution_time = time.perf_counter() - start_time
            
            return (execution_time,)
            
        except Exception as e:
            print(f"Error in decomposition evaluation: {e}")
            return (float('inf'),)
    
    def optimize(self):
        """Run island-based evolutionary optimization"""
        islands = [self.toolbox.population(n=self.pop_size) 
                  for _ in range(self.num_islands)]
        best_solutions = []
        
        for gen in range(self.num_generations):

            for i, island in enumerate(islands):
                # Evolve island population
                offspring = algorithms.varAnd(
                    island, self.toolbox, cxpb=0.7, mutpb=0.3)
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                
                island[:] = self.toolbox.select(offspring + island, k=len(island))
                
                # Store best solution
                best_ind = tools.selBest(island, k=1)[0]
                best_solutions.append((best_ind, best_ind.fitness.values[0]))
                
                print(f"Island {i}, Generation {gen}: Best fitness = {best_ind.fitness.values[0]}")
        
        # Return best overall solution
        best_solution = min(best_solutions, key=lambda x: x[1])
        
        # Update learner with best solution
        self.learner.update_model(
            self.topology_features,
            {
                'batch_size': int(best_solution[0] * 1024),
                'num_clusters': int(best_solution[1] * 20),
                'strategy': int(best_solution[2] * self.num_strategies)
            },
            best_solution.fitness.values[0]
        )
        
        return best_solution

def decompose_edges(edge_values, edge_sources, edge_targets, max_edges_per_batch, num_nodes, adjacency_matrix):
    """Enhanced edge decomposition with automatic strategy selection"""
    optimizer = DecompositionOptimizer(edge_values, edge_sources, edge_targets, num_nodes, adjacency_matrix)
    best_params = optimizer.optimize()
    
    # Use best parameters to perform final decomposition
    batch_size = int(best_params[0] * 1024)
    num_clusters = int(best_params[1] * 20)
    strategy = int(best_params[2] * 5)  # Now 5 strategies
    
    print(f"Best decomposition strategy: {strategy}")
    print(f"Best batch size: {batch_size}")
    print(f"Best number of clusters: {num_clusters}")
    
    if strategy == 0:
        return bfs_edge_decomposition(edge_values, edge_sources, edge_targets, batch_size)
    elif strategy == 1:
        return spectral_edge_decomposition(edge_values, edge_sources, edge_targets, 
                                         max(2, num_clusters))
    elif strategy == 2:
        return locality_aware_decomposition(edge_values, edge_sources, edge_targets, 
                                         batch_size)
    elif strategy == 3:
        return pagerank_edge_decomposition(edge_values, edge_sources, edge_targets,
                                         num_nodes, batch_size)
    else:
        return metis_style_decomposition(edge_values, edge_sources, edge_targets,
                                       num_nodes, max(2, num_clusters))

import threading
from contextlib import contextmanager
import threading

class CUDAContextManager:
    _instance = None
    _lock = threading.Lock()
    _context = None
    
    @classmethod
    def initialize(cls):
        with cls._lock:
            if cls._context is None:
                cls._context = cuda.Device(0).make_context()
    
    @classmethod
    def destroy(cls):
        with cls._lock:
            if cls._context is not None:
                cls._context.detach()
                cls._context = None
    
    @classmethod
    @contextmanager
    def get_context(cls):
        with cls._lock:
            if cls._context is None:
                cls.initialize()
            cls._context.push()
        try:
            yield
        finally:
            cuda.Context.pop()

class EdgePipeline:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.output_queue = Queue()
        self.lock = threading.Lock()
        self.active_workers = 0
        self.max_workers = 2  # Limit concurrent workers
        self.worker_lock = threading.Lock()
        self.feature_matrix_gpu = None
        self.feature_matrix_shape = None
        
    def initialize_feature_matrix(self, feature_matrix):
        """Initialize feature matrix on GPU once"""
        if sp.issparse(feature_matrix):
            feature_matrix = feature_matrix.toarray()
        feature_matrix = feature_matrix.astype(np.float32)
        self.feature_matrix_shape = feature_matrix.shape
        self.feature_matrix_gpu = cuda.mem_alloc(feature_matrix.nbytes)
        cuda.memcpy_htod(self.feature_matrix_gpu, feature_matrix)
        
    def cleanup(self):
        """Clean up GPU resources"""
        if self.feature_matrix_gpu is not None:
            self.feature_matrix_gpu.free()
            self.feature_matrix_gpu = None
        
    def process_edge_batch(self, batch_idx, edge_data, num_nodes):
        try:
            with CUDAContextManager.get_context():
                edge_values, edge_sources, edge_targets = edge_data
                temp_matrix = sp.csr_matrix(
                    (edge_values, (edge_sources, edge_targets)),
                    shape=(num_nodes, num_nodes)
                ).astype(np.float32)
                
                result = sparse_matrix_multiply_edge_centric(
                    temp_matrix, 
                    np.zeros(self.feature_matrix_shape, dtype=np.float32),  # Placeholder
                    self.batch_size,
                    feature_matrix_gpu=self.feature_matrix_gpu
                )
                
                with self.lock:
                    self.output_queue.put((batch_idx, result))
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
        finally:
            with self.worker_lock:
                self.active_workers -= 1
    
    def process_edges(self, edge_batches, feature_matrix, num_nodes):
        """Process edges with shared feature matrix"""
        try:
            with CUDAContextManager.get_context():
                self.initialize_feature_matrix(feature_matrix)
                
                results = {}
                current_batch = 0
                total_batches = len(edge_batches)

                while current_batch < total_batches or self.active_workers > 0:
                    while (current_batch < total_batches and 
                           self.active_workers < self.max_workers):
                        with self.worker_lock:
                            self.active_workers += 1
                            worker = threading.Thread(
                                target=self.process_edge_batch,
                                args=(current_batch, edge_batches[current_batch], num_nodes)
                            )
                            worker.start()
                            current_batch += 1

                    try:
                        batch_idx, result = self.output_queue.get(timeout=0.1)
                        results[batch_idx] = result
                    except Empty:
                        continue

                return results
        finally:
            self.cleanup()

# Main execution code
if __name__ == "__main__":
    try:
        CUDAContextManager.initialize()
        results = []
        for graph_info in graphs:
            index = graph_info["index"]
            name = graph_info["name"]
            graph_type = graph_info["type"]
            graph = graph_info["graph"]
            feature_matrix = graph_info["feature_matrix"]
            num_nodes = graph_info["num_nodes"]
            sparsity = graph_info["sparsity"]
            print(f"Testing graph {index}")

            # Setup memory monitoring
            free_mem, total_mem = cuda.mem_get_info()
            memory_idle = total_mem - free_mem
            stop_event = threading.Event()
            executor = ThreadPoolExecutor(max_workers=1)
            context = cuda.Device(0).make_context()
            memory_thread = executor.submit(memory_monitor, stop_event, context)

            # Create adjacency matrix
            adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for node in graph.nodes:
                for neighbor in graph.neighbors(node):
                    adjacency_matrix[node, neighbor] = 1.0
            adjacency_matrix = adjacency_matrix.tocsr()

            feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])
            time.sleep(0.5)  # Wait for memory thread to start

            try:
                # Time the decomposition phase
                decomp_start = time.perf_counter()
                edge_values, edge_sources, edge_targets = prepare_edge_lists(adjacency_matrix)
                
                # Try both pipelined and non-pipelined approaches
                use_pipeline = num_nodes > 1000  # Use pipeline for larger graphs
                
                if use_pipeline:
                    max_edges_per_batch = min(1024, len(edge_values) // 10)
                    edge_batches = decompose_edges(edge_values, edge_sources, edge_targets, max_edges_per_batch, num_nodes, adjacency_matrix)
                    decomp_time = time.perf_counter() - decomp_start

                    # Process with pipeline
                    with CUDAContextManager.get_context():
                        start_event = cuda.Event()
                        end_event = cuda.Event()
                        start_event.record()
                        
                        pipeline = EdgePipeline(batch_size=max_edges_per_batch)
                        batch_results = pipeline.process_edges(
                            edge_batches, feature_matrix, num_nodes)
                        
                        # Combine results
                        result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
                        for batch_result in batch_results.values():
                            result += batch_result
                        
                        end_event.record()
                        end_event.synchronize()
                        mult_time = start_event.time_till(end_event)
                else:
                    # Process entire graph at once
                    decomp_time = time.perf_counter() - decomp_start
                    
                    with CUDAContextManager.get_context():
                        start_event = cuda.Event()
                        end_event = cuda.Event()
                        start_event.record()
                        
                        result = sparse_matrix_multiply_edge_centric(
                            adjacency_matrix, feature_matrix.toarray())
                        
                        end_event.record()
                        end_event.synchronize()
                        mult_time = start_event.time_till(end_event)

                # Verify the result
                is_correct = verify_result(result, adjacency_matrix, feature_matrix)
                if not is_correct:
                    print(f"Graph {name} failed verification.")

                # Stop memory tracking and get results
                stop_event.set()
                peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2

                results.append({
                    "graph_index": index,
                    "graph_name": name,
                    "graph_type": graph_type,
                    "method": "pycuda_sparse_edge_throughput_auto" + ("_piped" if use_pipeline else ""),
                    "decomposition_time": decomp_time,
                    "multiplication_time": mult_time / 1000.0,  # Convert ms to seconds
                    "memory_peak_mb": peak_memory_usage,
                    "num_edges": len(edge_values),
                    "use_pipeline": use_pipeline,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_nodes": num_nodes,
                    "sparsity": sparsity,
                    "is_correct": is_correct
                })

            except cuda.LaunchError as e:
                print(f"CUDA launch failed: {e}")
                stop_event.set()
                continue
            except Exception as e:
                print(f"Error processing graph {name}: {e}")
                stop_event.set()
                continue
            finally:
                context.pop()

    finally:
        CUDAContextManager.destroy()

    # Save results
    if os.path.exists("gnn_results.json"):
        with open("gnn_results.json", "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # Update results
    for result in results:
        if any(r["graph_index"] == result["graph_index"] and 
               r["method"] == result["method"] for r in all_results):
            all_results = [r for r in all_results if not (
                r["graph_index"] == result["graph_index"] and 
                r["method"] == result["method"])]
        all_results.append(result)

    with open("gnn_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("Results have been saved to 'gnn_results.json'.")
