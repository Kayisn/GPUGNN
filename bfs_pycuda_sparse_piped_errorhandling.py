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

import pycuda.driver as cuda
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

def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)


# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, stream=None, num_warmup=2, num_test_runs=5):
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
                grid=grid_size,
                stream=stream
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
                grid=grid_size,
                stream=stream
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

import numpy as np
import scipy.sparse as sp
from collections import deque

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

import threading
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes the CUDA driver

class SingletonCUDAContextManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SingletonCUDAContextManager, cls).__new__(cls)
                    cls._instance.context = cuda.Device(0).make_context()
        return cls._instance

    def __enter__(self):
        self._lock.acquire()
        self.context.push()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.pop()
        self._lock.release()

def process_cluster(cluster_data):
    """Process a single cluster using sparse matrix multiplication"""
    with CUDAThreadManager.get_context():
        sub_adj, sub_feat, nodes_idx, all_nodes = cluster_data
        result, cluster_time, _ = sparse_matrix_multiply_pycuda(
            sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
        # Return the result and the corresponding node indices
        return result, nodes_idx, cluster_time

from concurrent.futures import ThreadPoolExecutor

def process_clusters_pipelined(cluster_data, num_workers=4):
    """Process clusters with a limited number of worker threads"""
    results = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(process_cluster, data): idx for idx, data in enumerate(cluster_data)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, nodes_idx, cluster_time = future.result()
                with lock:
                    results[idx] = (result, nodes_idx, cluster_time)
            except Exception as e:
                print(f"Error processing cluster {idx}: {e}")
                with lock:
                    results[idx] = None
    return results

import threading
from contextlib import contextmanager
import threading

class CUDAThreadManager:
    _local = threading.local()
    
    @classmethod
    def get_context(cls):
        return SingletonCUDAContextManager()

class Pipeline:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.lock = threading.Lock()
        
    def process_batch(self, batch_idx, batch_data):
        with CUDAThreadManager.get_context():
            results = []
            for idx, data in batch_data:
                try:
                    sub_adj, sub_feat, nodes_idx, all_nodes = data
                    result, timing, _ = sparse_matrix_multiply_pycuda(
                        sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
                    results.append((idx, result[:len(nodes_idx)], nodes_idx, timing))
                except Exception as e:
                    print(f"Error in batch {batch_idx}, cluster {idx}: {e}")
                    continue
            
            with self.lock:
                self.output_queue.put((batch_idx, results))

    def process_clusters(self, cluster_data, num_workers=2):
        # Split into batches
        batches = []
        current_batch = []
        
        for idx, data in enumerate(cluster_data):
            current_batch.append((idx, data))
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        # Process batches with workers
        active_workers = []
        results = {}
        
        for batch_idx, batch in enumerate(batches):
            # Start new worker
            worker = threading.Thread(
                target=self.process_batch,
                args=(batch_idx, batch)
            )
            worker.start()
            active_workers.append(worker)
            
            # Collect results from completed workers
            while len(active_workers) >= num_workers:
                try:
                    batch_idx, batch_results = self.output_queue.get_nowait()
                    for idx, result, nodes_idx, timing in batch_results:
                        results[idx] = (result, nodes_idx, timing)
                    active_workers = [w for w in active_workers if w.is_alive()]
                except Empty:  # Using the correct Empty exception
                    break
        
        # Wait for remaining workers
        for worker in active_workers:
            worker.join()
        
        # Collect remaining results
        while not self.output_queue.empty():
            batch_idx, batch_results = self.output_queue.get()
            for idx, result, nodes_idx, timing in batch_results:
                results[idx] = (result, nodes_idx, timing)
        
        return results

import networkx as nx
# Run tests and collect results
results = []
for graph_info in graphs:
    index = graph_info["index"]
    name = graph_info["name"]
    graph_type = graph_info["type"]
    if "graph" not in graph_info:
        print("Converting graph to nx")
        adjacency_matrix_csr = sp.csr_matrix(graph_info["adjacency"])
        # Convert to NetworkX graph using the updated function
        graph = nx.from_scipy_sparse_array(adjacency_matrix_csr)
        print("Converting graph to nx")
    else:
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

    num_nodes = graph.number_of_nodes()
    edges = np.array(graph.edges())
    row_indices = edges[:, 0]
    col_indices = edges[:, 1]
    data = np.ones(len(row_indices), dtype=np.float32)
    adjacency_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    # Break these up into smaller blocks for larger graphs using BFS

    adjacency_matrix = adjacency_matrix.tocsr()

    feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

    time.sleep(0.5)  # Wait for memory thread to start

    try:
        
        # Calculate batch size dynamically based on GPU SMs
        num_sms = get_num_sms()
        threads_per_sm = 1024  # Adjust based on your GPU architecture
        total_threads = num_sms * threads_per_sm
        threads_per_edge = 1  # Define based on kernel requirements
        batch_size = total_threads // threads_per_edge


        # Time the decomposition phase separately
        decomp_start = time.perf_counter()
        
        # Create clusters
        number_of_edges = adjacency_matrix.nnz
        max_edges_per_cluster = min(batch_size, number_of_edges )
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

        results.append({
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "pycuda_sparse_bfs_piped_errorhandling",
            "decomposition_time": decomp_time,
            "multiplication_time": mult_time / 1000.0,  # Convert ms to seconds
            "time_std": np.std(cluster_times) / 1000.0 if cluster_times else 0,
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