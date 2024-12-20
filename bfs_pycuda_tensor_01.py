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
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import numpy as np
from numpy import random as np_random  # Add this import

import networkx as nx

import concurrent

from verification import verify_result

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

import os

# Set CUDA compiler path before importing pycuda
#os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
#os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']

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



# CUDA kernel for tensor core operations
TENSOR_CORE_KERNEL = """
#include <mma.h>
using namespace nvcuda;

__global__ void sparse_matmul_tensor(
    float *A_data, int *A_indices, int *A_indptr,
    float *B_data, int *B_indices, int *B_indptr,
    float *C, int num_rows, int num_cols, int num_cols_B
) {
    // Define tile sizes for tensor cores
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Shared memory for the tiles
    __shared__ float a_tile[WMMA_M][WMMA_K];
    __shared__ float b_tile[WMMA_K][WMMA_N];
    
    // Initialize accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols_B) {
        float sum = 0;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];
        
        // Load and multiply using tensor cores where possible
        for (int idx = row_start; idx < row_end; idx += WMMA_K) {
            int k_elements = min(WMMA_K, row_end - idx);
            
            // Load tiles into shared memory
            if (threadIdx.x < k_elements) {
                a_tile[threadIdx.y][threadIdx.x] = A_data[idx + threadIdx.x];
            }
            
            __syncthreads();
            
            // Create matrix fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float> b_frag;
            
            // Load fragments
            wmma::load_matrix_sync(a_frag, &a_tile[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &b_tile[0][0], WMMA_N);
            
            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            
            __syncthreads();
        }
        
        // Store result
        C[row * num_cols_B + col] = sum + acc_frag.x[0];
    }
}
"""

# Add new constant for timeout duration (in seconds)
KERNEL_TIMEOUT = 3600  # Adjust this value based on your needs

# Define the PyCUDA-based sparse matrix multiplication method
def sparse_matrix_multiply_pycuda(A, B, num_warmup=2, num_test_runs=5):
    """Modified matrix multiplication function with tensor core support and timeout handling"""
    
    
    # Add size check
    if A.shape[0] * B.shape[1] > 1e7:  # Adjust threshold as needed
        # Split into smaller chunks for large matrices
        chunk_size = int(1e7 // B.shape[1])
        results = []
        timings = []
        
        for i in range(0, A.shape[0], chunk_size):
            end_idx = min(i + chunk_size, A.shape[0])
            A_chunk = A[i:end_idx]
            chunk_result, chunk_time, chunk_std = _sparse_matrix_multiply_chunk(
                A_chunk, B, num_warmup, num_test_runs)
            results.append(chunk_result)
            timings.append(chunk_time)
            
        return np.vstack(results), np.mean(timings), np.std(timings)
    else:
        return _sparse_matrix_multiply_chunk(A, B, num_warmup, num_test_runs)

def _sparse_matrix_multiply_chunk(A, B, num_warmup=2, num_test_runs=5):
    """Helper function to process a chunk of the matrix multiplication"""
    gpu_caps = get_gpu_capabilities()
    
    # Choose appropriate kernel based on GPU capabilities
    if gpu_caps['has_tensor_cores']:
        kernel_source = TENSOR_CORE_KERNEL
        kernel_name = "sparse_matmul_tensor"
    else:
        kernel_source = """
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
        kernel_name = "sparse_matmul"
    
    # Compile the selected kernel
    mod = SourceModule(kernel_source)
    sparse_matmul = mod.get_function(kernel_name)
    
    # Adjust block size for tensor cores if available
    if gpu_caps['has_tensor_cores']:
        block_size = (16, 16, 1)  # Optimal for tensor cores
    else:
        block_size = (16, 16, 1)  # Standard block size

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

    # Ensure A and B have valid shapes
    if A.shape[0] == 0 or A.shape[1] == 0 or B.shape[0] == 0 or B.shape[1] == 0:
        raise ValueError("Input matrices A and B must have non-zero dimensions.")

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

    grid_size = (
        int(np.ceil(B_csr.shape[1] / 16)),
        int(np.ceil(A_csr.shape[0] / 16)),
        1,
    )

    try:
        # Set timeout for CUDA operations
        #cuda.Context.get_current().set_limit(cuda.limit.TIMEOUT, int(KERNEL_TIMEOUT * 1000))  # milliseconds
        
        # Add synchronization points
        cuda.Context.synchronize()
        
        # Rest of the implementation remains the same, but add extra synchronization
        for _ in range(num_warmup):
            sparse_matmul(
                A_data_gpu, A_indices_gpu, A_indptr_gpu,
                B_data_gpu, B_indices_gpu, B_indptr_gpu,
                C_gpu, np.int32(A.shape[0]), np.int32(A.shape[1]),
                np.int32(B.shape[1]),
                block=block_size,
                grid=grid_size
            )
            cuda.Context.synchronize()  # Ensure warmup completion

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
            cuda.Context.synchronize()  # Ensure completion before next iteration

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

    except cuda.Error as e:
        if "timeout" in str(e).lower():
            print(f"CUDA kernel timed out after {KERNEL_TIMEOUT} seconds")
        raise
    finally:
        # Ensure cleanup happens even if timeout occurs
        try:
            cuda.Context.synchronize()
            A_data_gpu.free()
            A_indices_gpu.free()
            A_indptr_gpu.free()
            B_data_gpu.free()
            B_indices_gpu.free()
            B_indptr_gpu.free()
            C_gpu.free()
        except cuda.Error:
            pass


class CUDAContextManager:
    def __init__(self):
        self.context = None
        
    def __enter__(self):
        self.context = cuda.Device(0).make_context()
        return self
        
    def __exit__(self, *args):
        if self.context:
            self.context.pop()

# Modify process_cluster function to include timeout handling
def process_cluster(cluster_data):
    """Process a single cluster using sparse matrix multiplication with timeout handling"""
    with CUDAContextManager() as ctx:
        try:
            cuda.Context.get_current().set_limit(cuda.limit.TIMEOUT, int(KERNEL_TIMEOUT * 1000))
            sub_adj, sub_feat, nodes_idx, all_nodes = cluster_data
            result, cluster_time, _ = sparse_matrix_multiply_pycuda(
                sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
            return result[:len(nodes_idx)], cluster_time
        except cuda.Error as e:
            if "timeout" in str(e).lower():
                print("Cluster processing timed out, skipping...")
            return None, None

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
import multiprocessing

class CUDAThreadManager:
    _local = threading.local()
    
    @classmethod
    @contextmanager
    def get_context(cls):
        if not hasattr(cls._local, 'context_count'):
            cls._local.context_count = 0
            cls._local.context = cuda.Device(0).make_context()
        else:
            cls._local.context.push()

        try:
            yield
        finally:
            cls._local.context.pop()

def process_cluster_worker(args):
    idx, data = args
    try:
        # Each process creates its own context
        context = cuda.Device(0).make_context()
        try:
            sub_adj, sub_feat, nodes_idx, all_nodes = data

            # Check if sub_adj and sub_feat have valid shapes
            if sub_adj.shape[0] == 0 or sub_adj.shape[1] == 0 or sub_feat.shape[0] == 0 or sub_feat.shape[1] == 0:
                print(f"Skipping cluster {idx} due to empty submatrices.")
                return idx, None, None, None

            result, timing, _ = sparse_matrix_multiply_pycuda(
                sub_adj, sub_feat, num_warmup=0, num_test_runs=1)
            return idx, result[:len(nodes_idx)], nodes_idx, timing
        finally:
            context.pop()
    except Exception as e:
        print(f"Error processing cluster {idx}: {e}")
        return idx, None, None, None

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
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_cluster_worker, enumerate(cluster_data))
        result_dict = {}
        for idx, cluster_result, nodes_idx, timing in results:
            if cluster_result is not None:
                result_dict[idx] = (cluster_result, nodes_idx, timing)
            else:
                print(f"Cluster {idx} was skipped or failed; excluding from results.")
                result_dict[idx] = None
        return result_dict


from cuda_partition_01 import gpu_partition_graph
def create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters, feature_matrix):
    """
    Create clusters using GPU-accelerated METIS/BFS hybrid partitioning with fallback
    """
    try:
        # Pass feature matrix to gpu_partition_graph
        return gpu_partition_graph(adjacency_matrix, num_clusters, feature_matrix)
    except Exception as e:
        print(f"GPU partitioning failed: {e}")
        return None


if __name__ == '__main__':
    # Run tests and collect results
    results = []
    try:
        context = cuda.Device(0).make_context()
        
        for graph_info in graphs:
            index = graph_info["index"]
            name = graph_info["name"]
            graph_type = graph_info["type"]
            if "graph" not in graph_info:
                print("Converting graph to nx")
                adjacency_matrix = sp.csr_matrix(graph_info["adjacency"])
                # Convert to NetworkX graph using the updated function
                graph = nx.from_scipy_sparse_array(adjacency_matrix)
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


            adjacency_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for node in graph.nodes:
                for neighbor in graph.neighbors(node):
                    adjacency_matrix[node, neighbor] = 1.0


            # Break these up into smaller blocks for larger graphs using BFS

            adjacency_matrix = adjacency_matrix.tocsr()

            feature_matrix = sp.csr_matrix(graph_info["feature_matrix"])

            time.sleep(0.5)  # Wait for memory thread to start

            # Calculate batch size dynamically based on GPU SMs
            num_sms = get_num_sms()
            threads_per_sm = 1024  # Adjust based on your GPU architecture
            total_threads = num_sms * threads_per_sm
            threads_per_edge = 1  # Define based on kernel requirements
            batch_size = total_threads // threads_per_edge

            try:
                # Time the decomposition phase separately
                decomp_start = time.perf_counter()
                
                # Define the number of clusters - more conservative estimate
                cluster_size = max(int(np.sqrt(num_nodes)), batch_size)  # Increased minimum size
                num_clusters = max(2, num_nodes // cluster_size)
                
                # Update cluster creation call to include feature matrix
                cluster_data = create_clusters_metis_bfs_gpu(
                    adjacency_matrix, 
                    num_clusters,
                    feature_matrix
                )
                if not cluster_data:
                    raise RuntimeError("Failed to create clusters")
                    
                # Add size check and splitting for large clusters
                processed_cluster_data = []
                for sub_adj, sub_feat, nodes_idx, all_nodes in cluster_data:
                    if sub_adj.shape[0] * sub_feat.shape[1] > 1e7:  # Adjust threshold as needed
                        # Split large clusters into smaller chunks
                        chunk_size = max(2, int(1e7 // sub_feat.shape[1]))
                        for i in range(0, len(nodes_idx), chunk_size):
                            chunk_nodes = nodes_idx[i:i + chunk_size]
                            chunk_adj = sub_adj[i:i + chunk_size, :][:, i:i + chunk_size]
                            chunk_feat = sub_feat[i:i + chunk_size, :]
                            processed_cluster_data.append((chunk_adj, chunk_feat, chunk_nodes, all_nodes))
                    else:
                        processed_cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))
                
                cluster_data = processed_cluster_data  # Replace with processed chunks

                decomp_time = time.perf_counter() - decomp_start

                # Start timing just the multiplication phase
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
                    if cluster_result is None:
                        continue  # Skip failed clusters

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
                else:
                    # Stop memory tracking and get results
                    stop_event.set()
                    
                    results.append({
                        "graph_index": index,
                        "graph_name": name,
                        "graph_type": graph_type,
                        "method": "pycuda_sparse_bfs_tensor_01",
                        "decomposition_time": decomp_time,
                        "multiplication_time": mult_time / 1000.0,  # Convert ms to seconds
                        "time_std": np.std(cluster_times) / 1000.0 if cluster_times else 0,
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_nodes": num_nodes,
                        "sparsity": sparsity,
                        "is_correct": is_correct
                    })
                    # Modified second append to include total time
                    results.append({
                        "graph_index": index,
                        "graph_name": name,
                        "graph_type": graph_type,
                        "method": "pycuda_sparse_bfs_tensor_01_total_time",
                        "decomposition_time": 0,  # Set to 0 since we're using total time
                        "multiplication_time": (decomp_time + mult_time / 1000.0),  # Add decomp_time and mult_time
                        "time_std": np.std(cluster_times) / 1000.0 if cluster_times else 0,
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
                pass

    finally:
        context.pop()
        context.detach()  # Ensure complete cleanup

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
    with open("gnn_results.json", "w") as f:    json.dump(all_results, f, indent=4)# Print confirmationprint("Results have been saved to 'gnn_results.json'.")