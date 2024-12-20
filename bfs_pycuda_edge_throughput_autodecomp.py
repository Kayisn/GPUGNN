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

from queue import Queue
from threading import Lock
import threading
from concurrent.futures import ThreadPoolExecutor

def bfs_edge_decomposition(edge_values, edge_sources, edge_targets, max_edges_per_batch):
    """Thread-safe BFS-based edge decomposition strategy"""
    G = nx.Graph()
    for i in range(len(edge_sources)):
        G.add_edge(edge_sources[i], edge_targets[i], weight=edge_values[i])
    
    batches = []
    visited_edges = set()
    batch_lock = Lock()
    visited_lock = Lock()
    
    def process_node(start_node):
        edge_batch = ([], [], [])
        local_visited = set()
        
        # Local BFS traversal
        bfs_edges = list(nx.bfs_edges(G, start_node))
        for u, v in bfs_edges:
            with visited_lock:
                if (u, v) not in visited_edges and len(edge_batch[0]) < max_edges_per_batch:
                    edge_data = G.get_edge_data(u, v)
                    edge_batch[0].append(edge_data['weight'])
                    edge_batch[1].append(u)
                    edge_batch[2].append(v)
                    local_visited.add((u, v))
                    local_visited.add((v, u))
        
        if edge_batch[0]:
            with batch_lock:
                batches.append((
                    np.array(edge_batch[0], dtype=np.float32),
                    np.array(edge_batch[1], dtype=np.int32),
                    np.array(edge_batch[2], dtype=np.int32)
                ))
            with visited_lock:
                visited_edges.update(local_visited)
    
    # Process nodes in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_node, G.nodes())
    
    return batches

def spectral_edge_decomposition(edge_values, edge_sources, edge_targets, num_clusters):
    """Fast approximate spectral clustering using Nyström method"""
    num_nodes = max(max(edge_sources), max(edge_targets)) + 1
    
    # Use sparse matrix throughout
    adj_matrix = sp.csr_matrix(
        (edge_values, (edge_sources, edge_targets)), 
        shape=(num_nodes, num_nodes)
    )
    
    # Sample subset of nodes for Nyström approximation
    sample_size = min(1000, num_nodes)  # Limit sample size
    sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
    
    # Compute reduced affinity matrix
    sample_adj = adj_matrix[sample_indices][:, sample_indices].toarray()
    
    # Approximate eigenvectors using smaller matrix
    from scipy.sparse.linalg import eigsh
    try:
        _, sample_eigvecs = eigsh(sample_adj, k=num_clusters, which='LM')
    except:
        # Fallback to simpler clustering if eigendecomposition fails
        print("Warning: Eigendecomposition failed, using KMeans directly")
        clustering = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=1  # Faster, single init
        )
        node_clusters = clustering.fit_predict(adj_matrix.toarray())
    else:
        # Extend to full matrix using Nyström
        extension = adj_matrix[:, sample_indices].toarray() @ np.linalg.pinv(sample_adj)
        approx_eigvecs = extension @ sample_eigvecs
        
        # Quick KMeans on approximate eigenvectors
        clustering = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=1
        )
        node_clusters = clustering.fit_predict(approx_eigvecs)
    
    # Fast batch creation using numpy operations
    edge_array = np.column_stack([edge_values, edge_sources, edge_targets])
    batches = [[] for _ in range(num_clusters)]
    
    # Vectorized assignment
    src_clusters = node_clusters[edge_array[:, 1].astype(int)]
    for cluster in range(num_clusters):
        mask = src_clusters == cluster
        if np.any(mask):
            cluster_edges = edge_array[mask]
            batches[cluster] = (
                cluster_edges[:, 0].astype(np.float32),
                cluster_edges[:, 1].astype(np.int32),
                cluster_edges[:, 2].astype(np.int32)
            )
    
    return [batch for batch in batches if len(batch[0]) > 0]

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
    """Lock-free PageRank-based edge decomposition"""
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
    """Lock-free METIS-style partitioning"""
    # Use atomic operations for partition updates
    ATOMIC_PARTITION = """
    extern "C" __global__ void atomic_partition_update(
        int* node_partitions,
        float* partition_costs,
        const int* edge_sources,
        const int* edge_targets,
        const float* edge_weights,
        const int num_edges,
        const int num_partitions
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_edges) {
            int src = edge_sources[tid];
            int dst = edge_targets[tid];
            int src_part = node_partitions[src];
            int dst_part = node_partitions[dst];
            float weight = edge_weights[tid];
            
            atomicAdd(&partition_costs[src_part], weight);
            atomicAdd(&partition_costs[dst_part], weight);
        }
    }
    """
    
    # Initialize with thread-safe structures
    node_partitions = np.random.randint(0, num_partitions, num_nodes, dtype=np.int32)
    partition_counts = np.zeros(num_partitions, dtype=np.int32)
    
    mod = SourceModule(ATOMIC_PARTITION)
    partition_kernel = mod.get_function("atomic_partition_update")
    
    # Use CUDA events for synchronization
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    """GPU-accelerated METIS-style graph partitioning"""
    # Initialize random partitions
    node_partitions = np.random.randint(0, num_partitions, num_nodes, dtype=np.int32)
    
    # Allocate GPU memory
    node_partitions_gpu = cuda.mem_alloc(node_partitions.nbytes)
    partition_scores_gpu = cuda.mem_alloc(num_partitions * num_partitions * 4)  # float32
    
    # Iterative refinement
    max_iter = 20
    for _ in range(max_iter):
        cuda.memcpy_htod(node_partitions_gpu, node_partitions)
        cuda.memset_d32(partition_scores_gpu, 0, num_partitions * num_partitions)
        
        # Score partitions
        partition_kernel(
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

class DecompositionOptimizer:
    def __init__(self, edge_values, edge_sources, edge_targets, num_nodes, feature_matrix):  # Added feature_matrix parameter
        self.edge_values = edge_values
        self.edge_sources = edge_sources
        self.edge_targets = edge_targets
        self.num_nodes = num_nodes
        # Convert feature matrix to numpy array and ensure float32 type
        if sp.issparse(feature_matrix):
            self.feature_matrix = feature_matrix.toarray()
        else:
            self.feature_matrix = np.array(feature_matrix)
        self.feature_matrix = self.feature_matrix.astype(np.float32)
        
        # Define evolution parameters
        self.pop_size = 4  # Reduced population size
        self.num_generations = 3  # Reduced generations
        self.num_islands = 2  # Reduced islands
        
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
        
        # Update number of strategies to 5
        self.num_strategies = 1
        # Add timeout parameter
        self.optimization_timeout = 30  # seconds
        self.pipeline = EdgePipeline(batch_size=1024)  # Create single pipeline instance
        self.evaluation_timeout = 5  # Add timeout for individual evaluations
    
    def evaluate_decomposition(self, individual):
        """Thread-safe decomposition evaluation with timeout and better error handling"""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        def _evaluate():
            try:
                with CUDAContextManager.get_context():
                    batch_size = int(individual[0] * 1024)
                    num_clusters = int(individual[1] * 20)
                    strategy = int(individual[2] * self.num_strategies)
                    
                    # Choose decomposition strategy with timeout protection
                    if strategy == 0:
                        batches = locality_aware_decomposition(
                            self.edge_values, self.edge_sources, self.edge_targets, batch_size)
                    elif strategy == 1:
                        batches = spectral_edge_decomposition(
                            self.edge_values, self.edge_sources, self.edge_targets, 
                            max(2, num_clusters))
                    elif strategy == 2:
                        batches = bfs_edge_decomposition(
                            self.edge_values, self.edge_sources, self.edge_targets, batch_size)
                    elif strategy == 3:
                        batches = pagerank_edge_decomposition(
                            self.edge_values, self.edge_sources, self.edge_targets, 
                            self.num_nodes, batch_size)
                    else:
                        batches = metis_style_decomposition(
                            self.edge_values, self.edge_sources, self.edge_targets, 
                            self.num_nodes, max(2, num_clusters))
                    
                    start_time = time.perf_counter()
                    result = self.pipeline.process_edges(batches, self.feature_matrix, self.num_nodes)
                    execution_time = time.perf_counter() - start_time
                    
                    return execution_time
            except Exception as e:
                print(f"Evaluation error: {str(e)}")
                return float('inf')
        
        # Execute evaluation with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(_evaluate)
                result = future.result(timeout=self.evaluation_timeout)
                return (result,)
            except TimeoutError:
                print("Evaluation timeout reached")
                return (float('inf'),)
            except Exception as e:
                print(f"Evaluation failed: {str(e)}")
                return (float('inf'),)

    def optimize(self):
        """Run island-based evolutionary optimization with better timeout handling"""
        optimization_start = time.perf_counter()
        
        try:
            # Initialize population with pre-evaluated individuals
            islands = []
            for _ in range(self.num_islands):
                population = self.toolbox.population(n=self.pop_size)
                # Pre-evaluate initial population
                for ind in population:
                    try:
                        ind.fitness.values = self.evaluate_decomposition(ind)
                    except Exception as e:
                        print(f"Initial evaluation failed: {str(e)}")
                        ind.fitness.values = (float('inf'),)
                islands.append(population)
            
            best_solutions = []
            
            for gen in range(self.num_generations):

                for i, island in enumerate(islands):
                    # Evolve island population
                    offspring = algorithms.varAnd(
                        island, self.toolbox, cxpb=0.7, mutpb=0.3)
                    fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                    
                    for fit, ind in zip(fits, offspring):
                        ind.fitness.values = fit  # Assign fitness values to individuals
                        
                    island[:] = self.toolbox.select(offspring + island, k=len(island))
                    
                    # Store best solution
                    best_ind = tools.selBest(island, k=1)[0]
                    best_solutions.append((best_ind, best_ind.fitness.values[0]))
                    
                    print(f"Island {i}, Generation {gen}: Best fitness = {best_ind.fitness.values[0]}")

            if best_solutions:
                return min(best_solutions, key=lambda x: x[1])[0]
            else:
                return self.toolbox.individual()
            
            if time.perf_counter() - optimization_start > self.optimization_timeout:
                print("Optimization timeout reached, returning best solution so far")
                if best_solutions:
                    return min(best_solutions, key=lambda x: x[1])[0]
                return self.toolbox.individual()
                
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return self.toolbox.individual()

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.cleanup()

# Add a new class to represent batched data
class BatchData:
    def __init__(self, edge_values, edge_sources, edge_targets, node_indices, features):
        self.edge_values = edge_values
        self.edge_sources = edge_sources
        self.edge_targets = edge_targets
        self.node_indices = node_indices
        self.features = features

def create_feature_batches(feature_matrix, edge_batches):
    """Create matching feature matrix batches during decomposition"""
    if sp.issparse(feature_matrix):
        feature_matrix = feature_matrix.toarray()
    feature_matrix = feature_matrix.astype(np.float32)
    
    batched_data = []
    for edge_values, edge_sources, edge_targets in edge_batches:
        # Get unique nodes for this batch
        batch_nodes = set(np.unique(np.concatenate([edge_sources, edge_targets])))
        node_indices = np.array(list(batch_nodes))
        
        # Create node index mapping and remap edges
        node_map = {old: new for new, old in enumerate(node_indices)}
        local_sources = np.array([node_map[src] for src in edge_sources], dtype=np.int32)
        local_targets = np.array([node_map[dst] for dst in edge_targets], dtype=np.int32)
        
        # Extract features for batch nodes
        batch_features = feature_matrix[node_indices]
        
        batched_data.append(BatchData(
            edge_values,
            local_sources,
            local_targets,
            node_indices,
            batch_features
        ))
    
    return batched_data

def decompose_edges(edge_values, edge_sources, edge_targets, max_edges_per_batch, num_nodes, feature_matrix):  # Added feature_matrix parameter
    """Enhanced edge decomposition with automatic strategy selection"""
    optimizer = DecompositionOptimizer(
        edge_values, 
        edge_sources, 
        edge_targets, 
        num_nodes,
        feature_matrix  # Pass feature matrix to optimizer
    )
    best_params = optimizer.optimize()
    
    # Use best parameters to perform final decomposition
    batch_size = int(best_params[0] * 1024)
    num_clusters = int(best_params[1] * 20)
    strategy = int(best_params[2] * 1)  # Now 5 strategies
    
    print(f"Best decomposition strategy: {strategy}")
    print(f"Best batch size: {batch_size}")
    print(f"Best number of clusters: {num_clusters}")
    
    # Get edge batches using selected strategy
    edge_batches = None
    if strategy == 0:
        edge_batches = locality_aware_decomposition(edge_values, edge_sources, edge_targets, batch_size)
    elif strategy == 1:
        edge_batches = spectral_edge_decomposition(edge_values, edge_sources, edge_targets, max(2, num_clusters))
    elif strategy == 2:
        edge_batches = bfs_edge_decomposition(edge_values, edge_sources, edge_targets, batch_size)
    elif strategy == 3:
        edge_batches = pagerank_edge_decomposition(edge_values, edge_sources, edge_targets, num_nodes, batch_size)
    else:
        edge_batches = metis_style_decomposition(edge_values, edge_sources, edge_targets, num_nodes, max(2, num_clusters))
    
    # Create feature batches as part of decomposition
    return create_feature_batches(feature_matrix, edge_batches)

import threading
from contextlib import contextmanager
import threading

class CUDAContextManager:
    _instance = None
    _lock = threading.Lock()
    _context = None
    _ref_count = 0
    
    @classmethod
    def initialize(cls):
        with cls._lock:
            if cls._context is None:
                cls._context = cuda.Device(0).make_context()
                cls._ref_count = 0
    
    @classmethod
    def destroy(cls):
        with cls._lock:
            if cls._context is not None and cls._ref_count == 0:
                try:
                    cls._context.detach()
                except:
                    pass
                cls._context = None
    
    @classmethod
    @contextmanager
    def get_context(cls):
        with cls._lock:
            if cls._context is None:
                cls.initialize()
            cls._ref_count += 1
            try:
                cls._context.push()
            except:
                cls._ref_count -= 1
                raise
        try:
            yield
        finally:
            with cls._lock:
                try:
                    cls._context.pop()
                except Exception as e:
                    print(f"Error popping CUDA context: {e}")
                cls._ref_count -= 1
                if cls._ref_count == 0:
                    cls.destroy()

class SafeGPUMemory:
    def __init__(self, allocation_func, *args, **kwargs):
        self.ptr = None
        self.allocation_func = allocation_func
        self.args = args
        self.kwargs = kwargs
    
    def __enter__(self):
        try:
            self.ptr = self.allocation_func(*self.args, **self.kwargs)
            return self.ptr
        except Exception as e:
            print(f"GPU memory allocation failed: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ptr is not None:
            try:
                self.ptr.free()
            except:
                pass
            self.ptr = None

import multiprocessing

class EdgePipeline:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.output_queue = Queue()
        self.active_workers = multiprocessing.Value('i', 0)  # Thread-safe counter
        self.max_workers = 2
        self.worker_lock = threading.Lock()
        self.feature_matrix_gpu = None
        self.feature_matrix_shape = None
        self.feature_batches = None
        self._lock = threading.Lock()  # Add lock for thread safety
    
    def create_feature_batches(self, feature_matrix, edge_batches):
        """Create matching feature matrix batches based on edge batches"""
        if sp.issparse(feature_matrix):
            feature_matrix = feature_matrix.toarray()
        feature_matrix = feature_matrix.astype(np.float32)
        
        self.feature_batches = []
        unique_nodes = set()
        
        # For each edge batch, collect unique nodes involved
        for edge_values, edge_sources, edge_targets in edge_batches:
            batch_nodes = set(np.unique(np.concatenate([edge_sources, edge_targets])))
            unique_nodes.update(batch_nodes)
            
            # Create feature matrix slice for this batch's nodes
            node_indices = np.array(list(batch_nodes))
            batch_features = feature_matrix[node_indices]
            self.feature_batches.append((node_indices, batch_features))
        
        return len(unique_nodes)  # Return number of unique nodes for verification

    def process_edge_batch(self, batch_idx, edge_data, num_nodes):
        try:
            with CUDAContextManager.get_context():
                edge_values, edge_sources, edge_targets = edge_data
                node_indices, batch_features = self.feature_batches[batch_idx]
                
                # Create node index mapping for this batch
                node_map = {old: new for new, old in enumerate(node_indices)}
                
                # Remap edge indices to batch-local indices
                local_sources = np.array([node_map[src] for src in edge_sources], dtype=np.int32)
                local_targets = np.array([node_map[dst] for dst in edge_targets], dtype=np.int32)
                
                # Create local adjacency matrix
                local_matrix = sp.csr_matrix(
                    (edge_values, (local_sources, local_targets)),
                    shape=(len(node_indices), len(node_indices))
                ).astype(np.float32)
                
                # Allocate GPU memory for batch features
                with SafeGPUMemory(cuda.mem_alloc, batch_features.nbytes) as feature_matrix_gpu:
                    cuda.memcpy_htod(feature_matrix_gpu, batch_features)
                    
                    # Process batch
                    result = sparse_matrix_multiply_edge_centric(
                        local_matrix,
                        batch_features,
                        self.batch_size,
                        feature_matrix_gpu
                    )
                
                # Store result with original node indices for later combining
                self.output_queue.put((batch_idx, (node_indices, result)))
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
        finally:
            with self.active_workers.get_lock():
                self.active_workers.value -= 1

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
        
    def process_edges(self, edge_batches, feature_matrix, num_nodes):
        """Process edges with batched feature matrices"""
        with self._lock:  # Ensure only one thread uses pipeline at a time
            try:
                with CUDAContextManager.get_context():
                    # Create feature matrix batches
                    total_unique_nodes = self.create_feature_batches(feature_matrix, edge_batches)
                    
                    results = {}
                    current_batch = 0
                    total_batches = len(edge_batches)

                    while current_batch < total_batches or self.active_workers.value > 0:
                        while (current_batch < total_batches and 
                            self.active_workers.value < self.max_workers):
                            with self.worker_lock:
                                self.active_workers.value += 1
                                worker = threading.Thread(
                                    target=self.process_edge_batch,
                                    args=(current_batch, edge_batches[current_batch], num_nodes)
                                )
                                worker.start()
                                current_batch += 1

                        try:
                            batch_idx, (node_indices, result) = self.output_queue.get(timeout=0.1)
                            results[batch_idx] = (node_indices, result)
                        except Empty:
                            continue

                    # Combine results from all batches
                    final_result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
                    for node_indices, batch_result in results.values():
                        final_result[node_indices] += batch_result

                    return final_result

            finally:
                self.cleanup()
                self.feature_batches = None

# Main execution code
if __name__ == "__main__":
    try:
        CUDAContextManager.initialize()
        results = []
        
        # Add overall timeout
        max_time_per_graph = 300  # 5 minutes per graph
        
        for graph_info in graphs:
            graph_start_time = time.perf_counter()
            
            try:
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

                # Perform edge decomposition and processing
                edge_values, edge_sources, edge_targets = prepare_edge_lists(adjacency_matrix)
                optimizer = DecompositionOptimizer(
                    edge_values, edge_sources, edge_targets, num_nodes, feature_matrix)
                best_params = optimizer.optimize()
                
                # Use best parameters to perform final decomposition
                batch_size = int(best_params[0] * 1024)
                num_clusters = int(best_params[1] * 20)
                strategy = int(best_params[2] * 1)  # Now 5 strategies

                print(f"Best decomposition strategy: {strategy}")
                print(f"Best batch size: {batch_size}")
                print(f"Best number of clusters: {num_clusters}")

                # Get edge batches using selected strategy
                edge_batches = None
                if strategy == 0:
                    edge_batches = locality_aware_decomposition(edge_values, edge_sources, edge_targets, batch_size)
                elif strategy == 1:
                    edge_batches = spectral_edge_decomposition(edge_values, edge_sources, edge_targets, max(2, num_clusters))
                elif strategy == 2:
                    edge_batches = bfs_edge_decomposition(edge_values, edge_sources, edge_targets, batch_size)
                elif strategy == 3:
                    edge_batches = pagerank_edge_decomposition(edge_values, edge_sources, edge_targets, num_nodes, batch_size)
                else:
                    edge_batches = metis_style_decomposition(edge_values, edge_sources, edge_targets, num_nodes, max(2, num_clusters))

                # Create feature batches as part of decomposition
                feature_batches = create_feature_batches(feature_matrix, edge_batches)

                # Process edge batches using pipeline
                pipeline = EdgePipeline(batch_size=batch_size)
                result = pipeline.process_edges(edge_batches, feature_matrix, num_nodes)
                pipeline.cleanup()

            except cuda.LaunchError as e:
                print(f"CUDA launch error: {str(e)}")
            except Exception as e:
                print(f"Error processing graph {name}: {str(e)}")
            finally:
                # Cleanup
                stop_event.set()
                context.pop()
                executor.shutdown(wait=True)

            if time.perf_counter() - graph_start_time > max_time_per_graph:
                print(f"Timeout reached for graph {name}")
                continue

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, cleaning up...")
    except Exception as e:
        print(f"Global error: {str(e)}")
    finally:
        try:
            CUDAContextManager.destroy()
        except:
            pass
        # Save results if any
        if results:
            with open("gnn_results.json", "w") as f:
                json.dump(results, f)

            print("Results have been saved to 'gnn_results.json'.")
