import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import math
from collections import defaultdict

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule

class HierarchicalDecomposition:
    def __init__(self, matrix):
        self.matrix = matrix
        self.clusters = []

    def decompose(self):
        # Simple decomposition - divide matrix into chunks
        n = self.matrix.shape[0]
        chunk_size = 1000  # Adjust based on available memory
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            self.clusters.append({
                'indices': list(range(i, end)),
                'size': end - i
            })

    def get_cluster_batch(self, available_memory):
        # Simple implementation - return all clusters
        return self.clusters

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)


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

def get_optimal_block_size():
    """Determine optimal block size based on GPU capabilities."""
    device = cuda.Device(0)
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)
    
    # Calculate optimal block dimension (multiple of warp size)
    block_dim = int(math.sqrt(max_threads_per_block))
    block_dim = (block_dim // warp_size) * warp_size
    return (block_dim, block_dim)

def estimate_memory_requirements(A, B):
    """Estimate GPU memory requirements for matrix multiplication."""
    memory_needed = (
        A.data.nbytes +  # A_data
        A.indices.nbytes +  # A_indices
        A.indptr.nbytes +  # A_indptr
        B.data.nbytes +  # B_data
        B.indices.nbytes +  # B_indices
        B.indptr.nbytes +  # B_indptr
        A.shape[0] * B.shape[1] * 4  # Result matrix (float32)
    )
    free_mem, _ = cuda.mem_get_info()
    return memory_needed, free_mem

def decompose_graph(adjacency_matrix, feature_matrix, max_size):
    """Basic graph decomposition for memory management."""
    num_rows = adjacency_matrix.shape[0]
    if (num_rows <= max_size):
        return [(adjacency_matrix, feature_matrix)]
    
    num_parts = math.ceil(num_rows / max_size)
    subgraphs = []
    for i in range(num_parts):
        start_idx = i * max_size
        end_idx = min((i + 1) * max_size, num_rows)
        
        # Extract submatrices
        sub_adj = adjacency_matrix[start_idx:end_idx, :]
        sub_features = feature_matrix[start_idx:end_idx, :]
        
        subgraphs.append((sub_adj, sub_features))
    
    return subgraphs

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

    # Create CUDA stream
    stream = cuda.Stream()

    try:
        # Allocate GPU memory
        A_data_gpu = cuda.mem_alloc(A_data.nbytes)
        A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
        A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
        B_data_gpu = cuda.mem_alloc(B_data.nbytes)
        B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
        B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
        C_gpu = cuda.mem_alloc(A_csr.shape[0] * B_csr.shape[1] * np.float32().itemsize)

        # Safe memory transfer
        cuda.memcpy_htod(A_data_gpu, A_data)
        cuda.memcpy_htod(A_indices_gpu, A_indices)
        cuda.memcpy_htod(A_indptr_gpu, A_indptr)
        cuda.memcpy_htod(B_data_gpu, B_data)
        cuda.memcpy_htod(B_indices_gpu, B_indices)
        cuda.memcpy_htod(B_indptr_gpu, B_indptr)

        # Optimized CUDA kernel with binary search and shared memory
        mod = SourceModule(
            """
        __device__ inline int binary_search(const int* array, int left, int right, int target) {
            while (left <= right) {
                int mid = (left + right) >> 1;
                if (array[mid] == target) return mid;
                if (array[mid] < target) left = mid + 1;
                else right = mid - 1;
            }
            return -1;
        }

        __global__ void sparse_matmul(const float* __restrict__ A_data,
                                    const int* __restrict__ A_indices,
                                    const int* __restrict__ A_indptr,
                                    const float* __restrict__ B_data,
                                    const int* __restrict__ B_indices,
                                    const int* __restrict__ B_indptr,
                                    float* __restrict__ C,
                                    const int num_rows,
                                    const int num_cols,
                                    const int num_cols_B) {
            __shared__ float shared_sum[32][32];
            
            const int row = blockIdx.y * 32 + threadIdx.y;
            const int col = blockIdx.x * 32 + threadIdx.x;
            
            if (row < num_rows && col < num_cols_B) {
                float sum = 0.0f;
                const int row_start = A_indptr[row];
                const int row_end = A_indptr[row + 1];
                
                #pragma unroll 4
                for (int idx = row_start; idx < row_end; ++idx) {
                    const int k = A_indices[idx];
                    const float a_val = A_data[idx];
                    const int col_start = B_indptr[k];
                    const int col_end = B_indptr[k + 1];
                    
                    const int pos = binary_search(B_indices, col_start, col_end - 1, col);
                    if (pos != -1) {
                        sum += a_val * B_data[pos];
                    }
                }
                
                shared_sum[threadIdx.y][threadIdx.x] = sum;
                __syncthreads();
                
                if (threadIdx.x == 0) {
                    float final_sum = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        final_sum += shared_sum[threadIdx.y][i];
                    }
                    C[row * num_cols_B + col] = final_sum;
                }
            }
        }
        """
        )

        sparse_matmul = mod.get_function("sparse_matmul")
        block_size = (32, 32, 1)  # Optimized block size
        grid_size = (
            int(np.ceil(B_csr.shape[1] / 32)),
            int(np.ceil(A_csr.shape[0] / 32)),
            1,
        )

        # Warmup runs
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A_csr.shape[0]), 
                np.int32(A_csr.shape[1]),
                np.int32(B_csr.shape[1]),
                block=block_size,
                grid=grid_size,
                stream=stream
            )
            stream.synchronize()

        # Actual test runs with timing
        times = []
        for _ in range(num_test_runs):
            start = cuda.Event()
            end = cuda.Event()
            
            start.record(stream)
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A_csr.shape[0]), 
                np.int32(A_csr.shape[1]),
                np.int32(B_csr.shape[1]),
                block=block_size,
                grid=grid_size,
                stream=stream
            )
            end.record(stream)
            end.synchronize()
            
            times.append(start.time_till(end))

        # Safe memory transfer back
        C_dense = np.empty((A_csr.shape[0], B_csr.shape[1]), dtype=np.float32)
        cuda.memcpy_dtoh(C_dense, C_gpu)
        
        return C_dense, np.mean(times), np.std(times)

    finally:
        # Ensure GPU memory is always freed
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

def reorder_matrix_by_clusters(adjacency_matrix, feature_matrix, cluster_batch):
    """Reorder matrices according to clustering for better locality."""
    # Get new ordering from clusters
    new_order = []
    for cluster in cluster_batch:
        new_order.extend(cluster['indices'])
    
    # Create reverse mapping
    reverse_order = np.zeros_like(new_order)
    reverse_order[new_order] = np.arange(len(new_order))
    
    # Reorder matrices
    adj_reordered = adjacency_matrix[new_order][:, new_order]
    feat_reordered = feature_matrix[new_order]
    
    return adj_reordered, feat_reordered, reverse_order

def process_graph(graph_info):
    context = cuda.Device(0).make_context()
    try:
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


        # Create adjacency matrix using LIL format first
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:  # Only process if node has neighbors
                adjacency_matrix.rows[node] = neighbors
                adjacency_matrix.data[node] = [1.0] * len(neighbors)
        # Convert to CSR after construction is complete
        adjacency_matrix = adjacency_matrix.tocsr()

        feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

        try:
            # Add debug prints
            print(f"Matrix sizes - Adjacency: {adjacency_matrix.shape}, Features: {feature_matrix.shape}")
            print(f"Non-zero elements - Adjacency: {adjacency_matrix.nnz}, Features: {feature_matrix.nnz}")
            
            free_mem, total_mem = cuda.mem_get_info()
            print(f"Available GPU memory before decomposition: {free_mem/1024**2:.2f} MB")

            # Create hierarchical decomposition
            print("Starting hierarchical decomposition...")
            decomp = HierarchicalDecomposition(adjacency_matrix)
            decomp.decompose()
            print("Decomposition completed")
            
            # Get available GPU memory
            _, free_mem = cuda.mem_get_info()
            available_memory = free_mem * 0.8  # Leave 20% buffer
            print(f"Available memory for processing: {available_memory/1024**2:.2f} MB")
            
            # Get batch of clusters that fits in memory
            print("Getting cluster batch...")
            cluster_batch = decomp.get_cluster_batch(available_memory)
            print(f"Number of clusters in batch: {len(cluster_batch)}")
            
            results_by_level = defaultdict(list)
            
            # Initialize final result as LIL format for efficient updates
            final_result = sp.lil_matrix(feature_matrix.shape, dtype=np.float32)
            
            # Start memory tracking
            free_mem_start, total_mem = cuda.mem_get_info()
            memory_idle = total_mem - free_mem_start
            stop_event = threading.Event()
            memory_thread = executor.submit(memory_monitor, stop_event, context)
            
            # Reorder matrices based on clustering
            print("Reordering matrices...")
            adj_reordered, feat_reordered, reverse_order = reorder_matrix_by_clusters(
                adjacency_matrix, feature_matrix, cluster_batch
            )
            
            # Perform single sparse matrix multiplication with reordered matrices
            print("Performing sparse matrix multiplication...")
            result_reordered, avg_time, std_time = sparse_matrix_multiply_pycuda(
                adj_reordered, feat_reordered,
                num_warmup=2,
                num_test_runs=5
            )
            
            # Restore original ordering
            final_result = result_reordered[reverse_order]
                
            # Stop memory tracking and get peak usage
            stop_event.set()
            peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2
                
            # Store timing results
            results.append({
                "graph_index": index,
                "graph_name": name,
                "graph_type": graph_type,
                "method": "pycuda_sparse_decomposed",
                "time_seconds": avg_time / 1000.0,  # Convert ms to seconds
                "time_std": std_time / 1000.0,  # Convert ms to seconds
                "memory_peak_mb": peak_memory_usage,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_nodes": num_nodes,
                "sparsity": sparsity,
            })
            
            print(f"\nProcessing completed successfully:")
            print(f"- Average time: {avg_time:.2f}ms")
            print(f"- Peak memory usage: {peak_memory_usage:.2f}MB")
            
            # Convert final result to dense array for output
            final_dense = final_result.toarray()
                    
        except Exception as e:
            print(f"Error processing graph {graph_info['name']}: {str(e)}")
            print(f"Exception details:", e.__class__.__name__)
            if 'stop_event' in locals():
                stop_event.set()
        

        context.pop()
        
        return final_dense, avg_time, std_time, peak_memory_usage

    finally:
        context.pop()

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

    # Setup memory tracking
    free_mem, total_mem = cuda.mem_get_info()
    memory_idle = total_mem - free_mem
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    context = cuda.Device(0).make_context()
    memory_thread = executor.submit(memory_monitor, stop_event, context)

    try:
        # Create adjacency matrix
        adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                adjacency_matrix[node, neighbor] = 1.0
        adjacency_matrix = adjacency_matrix.tocsr()
        feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

        # Debug prints
        print(f"Matrix sizes - Adjacency: {adjacency_matrix.shape}, Features: {feature_matrix.shape}")
        print(f"Non-zero elements - Adjacency: {adjacency_matrix.nnz}, Features: {feature_matrix.nnz}")
        
        # Create hierarchical decomposition
        decomp = HierarchicalDecomposition(adjacency_matrix)
        decomp.decompose()
        
        # Get cluster batch that fits in memory
        _, free_mem = cuda.mem_get_info()
        available_memory = free_mem * 0.8  # Leave 20% buffer
        cluster_batch = decomp.get_cluster_batch(available_memory)
        
        # Reorder matrices based on clustering
        adj_reordered, feat_reordered, reverse_order = reorder_matrix_by_clusters(
            adjacency_matrix, feature_matrix, cluster_batch
        )
        
        # Execute computation
        result, avg_time, std_time = sparse_matrix_multiply_pycuda(
            adj_reordered, 
            feat_reordered,
            num_warmup=2,
            num_test_runs=5
        )
        
        # Restore original ordering
        final_result = result[reverse_order]
        
        # Stop memory tracking and get results
        stop_event.set()
        peak_memory_usage = (memory_thread.result() - memory_idle) / 1024**2

        results.append({
            "graph_index": index,
            "graph_name": name,
            "graph_type": graph_type,
            "method": "pycuda_sparse_decomposed",
            "time_seconds": avg_time / 1000.0,  # Convert ms to seconds
            "time_std": std_time / 1000.0,
            "memory_peak_mb": peak_memory_usage,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_nodes": num_nodes,
            "sparsity": sparsity,
        })

        print(f"\nProcessing completed successfully:")
        print(f"- Average time: {avg_time:.2f}ms")
        print(f"- Peak memory usage: {peak_memory_usage:.2f}MB")

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