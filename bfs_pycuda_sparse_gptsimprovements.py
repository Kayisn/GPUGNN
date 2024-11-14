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

import concurrent

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

# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, num_warmup=2, num_test_runs=5):
    # Ensure A and B are in CSR format
    A_csr = A.tocsr().astype(np.float32)
    B_csr = B.tocsr().astype(np.float32)

    # Extract CSR components
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr

    B_data = B_csr.data
    B_indices = B_csr.indices
    B_indptr = B_csr.indptr

    # Allocate GPU memory for CSR components
    A_data_gpu = cuda.mem_alloc(A_data.nbytes)
    A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
    A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
    B_data_gpu = cuda.mem_alloc(B_data.nbytes)
    B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
    B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
    C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * A_data.dtype.itemsize)

    # Copy data to GPU (synchronous)
    cuda.memcpy_htod(A_data_gpu, A_data)
    cuda.memcpy_htod(A_indices_gpu, A_indices)
    cuda.memcpy_htod(A_indptr_gpu, A_indptr)
    cuda.memcpy_htod(B_data_gpu, B_data)
    cuda.memcpy_htod(B_indices_gpu, B_indices)
    cuda.memcpy_htod(B_indptr_gpu, B_indptr)

    # CUDA kernel for sparse matrix multiplication
    mod = SourceModule(
        """
    __global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, float *B_data, int *B_indices, int *B_indptr, float *C, int num_rows, int num_cols, int num_cols_B) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < num_rows && col < num_cols_B) {
            float sum = 0;
            int row_start = A_indptr[row];
            int row_end = A_indptr[row + 1];
            for (int idx = row_start; idx < row_end; ++idx) {
                int k = A_indices[idx];
                int col_start = B_indptr[k];
                int col_end = B_indptr[k + 1];
                for (int jdx = col_start; jdx < col_end; ++jdx) {
                    if (B_indices[jdx] == col) {
                        sum += A_data[idx] * B_data[jdx];
                        break;
                    }
                }
            }
            C[row * num_cols_B + col] = sum;
        }
    }
    """
    )

    sparse_matmul = mod.get_function("sparse_matmul")
    block_size = (16, 16, 1)
    grid_size = (
        int(np.ceil(B_csr.shape[1] / 16)),
        int(np.ceil(A_csr.shape[0] / 16)),
        1,
    )

    try:
        # Synchronous execution
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            cuda.Context.synchronize()

        # Timing runs
        start_event = cuda.Event()
        end_event = cuda.Event()
        
        times = []
        for _ in range(num_test_runs):
            start_event.record()
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            end_event.record()
            end_event.synchronize()
            times.append(start_event.time_till(end_event))

        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Synchronous copy back
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)

        # Free GPU memory
        A_data_gpu.free()
        A_indices_gpu.free()
        A_indptr_gpu.free()
        B_data_gpu.free()
        B_indices_gpu.free()
        B_indptr_gpu.free()
        C_gpu.free()

        return C_dense, mean_time, std_time

    except:
        # Cleanup on error
        try:
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()
        except:
            pass
        raise

def bfs_cluster(adjacency_matrix, start_node, max_edges):
    """Create a cluster using BFS until reaching max_edges"""
    nodes_in_cluster = set()
    edges_in_cluster = 0
    queue = [start_node]
    nodes_in_cluster.add(start_node)
    
    while queue and edges_in_cluster < max_edges:
        current = queue.pop(0)
        row = adjacency_matrix[current].tocsr()
        neighbors = row.indices
        
        for neighbor in neighbors:
            if neighbor not in nodes_in_cluster:
                nodes_in_cluster.add(neighbor)
                queue.append(neighbor)
            edges_in_cluster += 1  # Count all edges, even to existing nodes
            
    return list(nodes_in_cluster)

def create_clusters(adjacency_matrix, max_edges_per_cluster):
    """Divide the graph into clusters ensuring all edges are covered"""
    num_nodes = adjacency_matrix.shape[0]
    remaining_edges = set((i, j) for i, j in zip(*adjacency_matrix.nonzero()))
    unvisited = set(range(num_nodes))
    clusters = []
    
    while unvisited:
        start_node = min(unvisited)  # Use deterministic node selection
        cluster_nodes = bfs_cluster(adjacency_matrix, start_node, max_edges_per_cluster)
        clusters.append(cluster_nodes)
        
        # Remove processed edges from remaining_edges
        for node in cluster_nodes:
            row = adjacency_matrix[node].tocsr()
            for neighbor in row.indices:
                remaining_edges.discard((node, neighbor))
        
        unvisited -= set(cluster_nodes)
    
    # Handle any remaining edges by adding their nodes to appropriate clusters
    while remaining_edges:
        edge = remaining_edges.pop()
        added = False
        for cluster in clusters:
            if edge[0] in cluster or edge[1] in cluster:
                cluster.extend([edge[0], edge[1]])
                cluster = list(set(cluster))  # Remove duplicates
                added = True
                break
        if not added:
            clusters.append([edge[0], edge[1]])
    
    return clusters

def extract_submatrices(adjacency_matrix, feature_matrix, cluster_nodes):
    """Extract submatrices including all necessary connections"""
    nodes_idx = sorted(cluster_nodes)
    
    # Get all nodes that are connected to the cluster nodes
    connected_nodes = set()
    for node in nodes_idx:
        row = adjacency_matrix[node].tocsr()
        connected_nodes.update(row.indices)
    
    # Include both cluster nodes and their neighbors
    all_nodes = sorted(set(nodes_idx) | connected_nodes)
    
    # Extract the relevant submatrices
    sub_adj = adjacency_matrix[nodes_idx, :][:, all_nodes]
    sub_feat = feature_matrix[all_nodes, :]
    
    return sub_adj, sub_feat, nodes_idx, all_nodes

class CUDAContextManager:
    def __init__(self):
        self.context = None
        
    def __enter__(self):
        self.context = cuda.Device(0).make_context()
        return self
        
    def __exit__(self, *args):
        if self.context:
            self.context.pop()

def process_cluster(cluster_data):
    """Process a single cluster using sparse matrix multiplication"""
    with CUDAContextManager():
        sub_adj, sub_feat, nodes_idx, all_nodes = cluster_data
        result, cluster_time, _ = sparse_matrix_multiply_pycuda(sub_adj, sub_feat, num_warmup=1, num_test_runs=1)
        # Only return results for the original cluster nodes
        return result[:len(nodes_idx)], cluster_time

def process_cluster_pipelined(cluster_queue, result_dict, lock):
    """Process clusters sequentially with proper context management"""
    with CUDAContextManager():
        while True:
            try:
                idx, (sub_adj, sub_feat, nodes_idx, all_nodes) = cluster_queue.get_nowait()
                try:
                    result, cluster_time, _ = sparse_matrix_multiply_pycuda(
                        sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                    
                    with lock:
                        result_dict[idx] = (result[:len(nodes_idx)], nodes_idx, cluster_time)
                except Exception as e:
                    print(f"Error processing cluster {idx}: {e}")
                    with lock:
                        result_dict[idx] = None
            except Empty:
                break

import threading
from contextlib import contextmanager
import threading

class CUDAThreadManager:
    _local = threading.local()
    
    @classmethod
    @contextmanager
    def get_context(cls):
        if not hasattr(cls._local, 'context_count'):
            cls._local.context_count = 0
        
        if cls._local.context_count == 0:
            ctx = cuda.Device(0).make_context()
        else:
            ctx = cuda.Context.get_current()
            ctx.push()
            
        cls._local.context_count += 1
        try:
            yield
        finally:
            cls._local.context_count -= 1
            cuda.Context.pop()
            if cls._local.context_count == 0:
                ctx.detach()

class Pipeline:
    def __init__(self, batch_size=None):
        # Auto-tune batch size based on available memory
        if batch_size is None:
            free_mem, total_mem = cuda.mem_get_info()
            self.batch_size = max(1, min(4, free_mem // (1024**3)))  # 1 batch per GB
        else:
            self.batch_size = batch_size
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.lock = threading.Lock()
        
    def process_batch(self, batch_idx, batch_data):
        with CUDAThreadManager.get_context():
            results = []
            # Pre-allocate CUDA events for better timing
            start_event = cuda.Event()
            end_event = cuda.Event()
            
            for idx, data in batch_data:
                try:
                    sub_adj, sub_feat, nodes_idx, all_nodes = data
                    start_event.record()
                    result, _, _ = sparse_matrix_multiply_pycuda(
                        sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                    end_event.record()
                    end_event.synchronize()
                    timing = start_event.time_till(end_event)
                    results.append((idx, result[:len(nodes_idx)], nodes_idx, timing))
                except Exception as e:
                    print(f"Error in batch {batch_idx}, cluster {idx}: {e}")
                    continue
            
            self.output_queue.put((batch_idx, results))

    def process_clusters(self, cluster_data, num_workers=2):
        # Pre-process batches
        batches = [cluster_data[i:i + self.batch_size] 
                  for i in range(0, len(cluster_data), self.batch_size)]
        
        # Process batches with workers
        results = {}
        active_workers = []
        
        for batch_idx, batch in enumerate(batches):
            batch_with_idx = list(enumerate(batch, start=batch_idx * self.batch_size))
            worker = threading.Thread(
                target=self.process_batch,
                args=(batch_idx, batch_with_idx)
            )
            worker.start()
            active_workers.append(worker)
            
            # Limit concurrent workers
            while len(active_workers) >= num_workers:
                self._collect_results(results, active_workers)
                
        # Wait for remaining workers
        while active_workers:
            self._collect_results(results, active_workers)
            
        return results
    
    def _collect_results(self, results, active_workers):
        try:
            batch_idx, batch_results = self.output_queue.get(timeout=0.1)
            for idx, result, nodes_idx, timing in batch_results:
                results[idx] = (result, nodes_idx, timing)
        except Empty:
            pass
        active_workers[:] = [w for w in active_workers if w.is_alive()]

# Run tests and collect results
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

    # Perform multiplication (example using BFS and feature matrix)
    aggregated_feature_matrix = feature_matrix.copy()

    free_mem, total_mem = cuda.mem_get_info()
    memory_idle = total_mem - free_mem
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    context = cuda.Device(0).make_context()

    memory_thread = executor.submit(memory_monitor, stop_event, context)


    adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            adjacency_matrix[node, neighbor] = 1.0


    # Break these up into smaller blocks for larger graphs using BFS

    adjacency_matrix = adjacency_matrix.tocsr()

    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    time.sleep(0.5)  # Wait for memory thread to start

    try:
        # Time the decomposition phase separately
        decomp_start = time.perf_counter()
        
        # Create clusters
        number_of_edges = adjacency_matrix.nnz
        max_edges_per_cluster = np.sqrt(number_of_edges)
        clusters = create_clusters(adjacency_matrix, max_edges_per_cluster)
        
        # Prepare cluster data
        cluster_data = []
        for cluster_nodes in clusters:
            sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices(
                adjacency_matrix, feature_matrix, cluster_nodes)
            cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))
        
        decomp_time = time.perf_counter() - decomp_start

        # Start timing just the multiplication phase
        with CUDAThreadManager.get_context():
            start_event = cuda.Event()
            end_event = cuda.Event()
            start_event.record()
            
            pipeline = Pipeline(batch_size=4)
            result_dict = pipeline.process_clusters(cluster_data, num_workers=2)
            
            # Combine results with proper error handling
            result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
            node_counts = np.zeros(num_nodes, dtype=np.int32)
            cluster_times = []
            
            success = False
            for idx in range(len(cluster_data)):
                if idx not in result_dict or result_dict[idx] is None:
                    continue
                    
                cluster_result, nodes_idx, cluster_time = result_dict[idx]
                result[nodes_idx] += cluster_result
                node_counts[nodes_idx] += 1
                cluster_times.append(cluster_time)
                success = True
            
            if not success:
                raise RuntimeError("All clusters failed to process")

            # Average results for overlapping nodes
            mask = node_counts != 0
            result[mask] = result[mask] / node_counts[mask, np.newaxis]

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
            "method": "pycuda_sparse_bfs_gptsuggestions",
            "decomposition_time": decomp_time,
            "multiplication_time": mult_time / 1000.0,  # Convert ms to seconds
            "time_std": np.std(cluster_times) / 1000.0 if cluster_times else 0,
            "memory_peak_mb": peak_memory_usage,
            "num_clusters": len(clusters),
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

import os

# Load existing results or create a new one
if os.path.exists("gnn_results.json"):
    with open("gnn_results.json", "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            # Initialize as an empty list if the file is empty or corrupted
            all_results = []
else:
    all_results = []

# Update results by replacing existing ones by graph index and method
for result in results:
    # Check if the result already exists in the list
    if any(
        r["graph_index"] == result["graph_index"] and r["method"] == result["method"]
        for r in all_results
    ):
        # If so, replace the existing result
        all_results = [
            r
            for r in all_results
            if not (
                r["graph_index"] == result["graph_index"]
                and r["method"] == result["method"]
            )
        ]
        all_results.append(result)
    else:
        all_results.append(result)

# Save results
with open("gnn_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

# Print confirmation
print("Results have been saved to 'gnn_results.json'.")
