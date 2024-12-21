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

gpu_caps = get_gpu_capabilities()
kernel_source = """
__global__ void sparse_matmul(float *A_data, int *A_indices, int *A_indptr, 
                            float *B_data, int *B_indices, int *B_indptr, 
                            float *C, int num_rows, int num_cols, int num_cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows && col < num_cols_B) {
        float sum = 0.0f;
        int row_start = A_indptr[row];
        int row_end = A_indptr[row + 1];
        
        // Coalesced memory access pattern for Maxwell
        for (int idx = row_start; idx < row_end; ++idx) {
            int k = A_indices[idx];
            float a_val = A_data[idx];
            
            int col_start = B_indptr[k];
            int col_end = B_indptr[k + 1];
            
            // Binary search for matching column
            int left = col_start;
            int right = col_end - 1;
            
            while (left <= right) {
                int mid = (left + right) >> 1;
                int bcol = B_indices[mid];
                if (bcol == col) {
                    sum += a_val * B_data[mid];
                    break;
                }
                if (bcol < col) left = mid + 1;
                else right = mid - 1;
            }
        }
        C[row * num_cols_B + col] = sum;
    }
}
"""
kernel_name = "sparse_matmul"

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
    
    if not nodes_idx or not all_nodes:
        return None, None, None, None
    
    # Ensure sub_adj and sub_feat are in CSR format
    sub_adj = sub_adj.tocsr()
    sub_feat = sub_feat.tocsr()
    return sub_adj, sub_feat, nodes_idx, all_nodes


def create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters):
    """
    Create clusters using GPU-accelerated METIS/BFS hybrid partitioning
    """
    from cuda_partition_simplified import gpu_partition_graph
    return gpu_partition_graph(adjacency_matrix, num_clusters)

class CUDAKernelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.kernel = None
        self.streams = {}
        self.context = None
        self.mod = None
    
    def init_context(self):
        if self.context is None:
            self.context = cuda.Device(0).make_context()
            # Compile kernel once
            try:
                self.mod = SourceModule(kernel_source)
                self.kernel = self.mod.get_function(kernel_name)
                if self.kernel is None:
                    raise RuntimeError("Failed to compile kernel")
            except Exception as e:
                print(f"Error compiling kernel: {e}")
                if self.context:
                    self.context.pop()
                    self.context = None
                raise
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_stream(self, thread_id=None):
        if thread_id is None:
            thread_id = threading.get_ident()
        if thread_id not in self.streams:
            self.streams[thread_id] = cuda.Stream()
        return self.streams[thread_id]
    
    def cleanup(self):
        if self.context:
            self.context.pop()
            self.context = None

class GPUPipeline:
    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.kernel_manager = CUDAKernelManager.get_instance()
        self.queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.block_size = (16, 16, 1)  # Default block size
        self.allocated_memory = {}
        self.ctx = None
        self.device = None
        self.use_async = False  # Flag for async operations
        self.kernel_compiled = False
        self.max_threads_per_block = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        self.block_size = None  # Will be set dynamically
        self.block = (16, 16, 1)  # Fixed block size for Maxwell
        self.mod = None
        self.kernel = None
        try:
            # Check if async operations are available
            self.use_async = hasattr(cuda, 'mem_alloc_async')
        except:
            self.use_async = False
    
    def init_kernel_manager(self):
        """Initialize kernel manager in the correct context"""
        if not self.kernel_compiled:
            try:
                if self.kernel_manager is None:
                    self.kernel_manager = CUDAKernelManager.get_instance()
                self.kernel_manager.init_context()
                if self.kernel_manager.kernel is None:
                    raise RuntimeError("Kernel compilation failed")
                self.kernel_compiled = True
            except Exception as e:
                print(f"Failed to initialize kernel manager: {e}")
                raise
        
    def calculate_grid_size(self, rows, cols):
        """Calculate appropriate grid and block sizes based on matrix dimensions and device limits"""
        max_block_size = int(np.sqrt(self.max_threads_per_block))
        block_x = min(32, max_block_size, cols)  # Typically 32 for coalesced memory access
        block_y = min(32, max_block_size, rows)
        
        self.block_size = (block_x, block_y, 1)
        
        grid_x = int(np.ceil(cols / block_x))
        grid_y = int(np.ceil(rows / block_y))
        
        return (grid_x, grid_y, 1)
    
    def allocate_gpu_memory(self, stream, *arrays):
        """Allocate GPU memory for multiple arrays with fallback to sync operations"""
        gpu_arrays = []
        try:
            for arr in arrays:
                if self.use_async:
                    gpu_arr = cuda.mem_alloc_async(arr.nbytes, stream)
                    cuda.memcpy_htod_async(gpu_arr, arr, stream)
                else:
                    gpu_arr = cuda.mem_alloc(arr.nbytes)
                    cuda.memcpy_htod(gpu_arr, arr)
                gpu_arrays.append(gpu_arr)
                self.allocated_memory[id(gpu_arr)] = gpu_arr
            return gpu_arrays
        except Exception as e:
            print(f"Memory allocation failed: {e}")
            self.free_gpu_memory(*gpu_arrays)
            return []

    def free_gpu_memory(self, *gpu_arrays):
        """Free GPU memory for multiple arrays"""
        for gpu_arr in gpu_arrays:
            if id(gpu_arr) in self.allocated_memory:
                try:
                    gpu_arr.free()
                    del self.allocated_memory[id(gpu_arr)]
                except cuda.Error:
                    pass

    def init_cuda(self):
        """Initialize CUDA context and device with proper kernel loading"""
        if self.ctx is None:
            cuda.init()
            self.device = cuda.Device(0)
            self.ctx = self.device.make_context()
            
            # Load kernel in this context
            self.mod = SourceModule(kernel_source)
            self.kernel = self.mod.get_function(kernel_name)
            
            # Set block size for Maxwell architecture
            self.block = (16, 16, 1)
    
    def process_batch(self, batch_data):
        """Process a batch using proper CUDA memory management"""
        try:
            self.init_cuda()
            
            results = []
            for idx, (sub_adj, sub_feat, nodes_idx, all_nodes) in batch_data:
                try:
                    # Convert to CSR and proper types
                    sub_adj = sub_adj.tocsr().astype(np.float32)
                    sub_feat = sub_feat.tocsr().astype(np.float32)

                    # Extract CSR components
                    A_data = sub_adj.data
                    A_indices = sub_adj.indices
                    A_indptr = sub_adj.indptr
                    B_data = sub_feat.data
                    B_indices = sub_feat.indices
                    B_indptr = sub_feat.indptr

                    # Allocate GPU memory in current context
                    gpu_ptrs = []
                    try:
                        A_data_gpu = cuda.mem_alloc(A_data.nbytes)
                        gpu_ptrs.append(A_data_gpu)
                        A_indices_gpu = cuda.mem_alloc(A_indices.nbytes)
                        gpu_ptrs.append(A_indices_gpu)
                        A_indptr_gpu = cuda.mem_alloc(A_indptr.nbytes)
                        gpu_ptrs.append(A_indptr_gpu)
                        B_data_gpu = cuda.mem_alloc(B_data.nbytes)
                        gpu_ptrs.append(B_data_gpu)
                        B_indices_gpu = cuda.mem_alloc(B_indices.nbytes)
                        gpu_ptrs.append(B_indices_gpu)
                        B_indptr_gpu = cuda.mem_alloc(B_indptr.nbytes)
                        gpu_ptrs.append(B_indptr_gpu)
                        
                        result_size = sub_adj.shape[0] * sub_feat.shape[1]
                        C_gpu = cuda.mem_alloc(result_size * np.float32().itemsize)
                        gpu_ptrs.append(C_gpu)

                        # Copy data to GPU in current context
                        cuda.memcpy_htod(A_data_gpu, A_data)
                        cuda.memcpy_htod(A_indices_gpu, A_indices)
                        cuda.memcpy_htod(A_indptr_gpu, A_indptr)
                        cuda.memcpy_htod(B_data_gpu, B_data)
                        cuda.memcpy_htod(B_indices_gpu, B_indices)
                        cuda.memcpy_htod(B_indptr_gpu, B_indptr)

                        # Calculate grid size
                        grid_x = int(np.ceil(sub_feat.shape[1] / self.block[0]))
                        grid_y = int(np.ceil(sub_adj.shape[0] / self.block[1]))
                        grid = (grid_x, grid_y)

                        # Add timing events
                        start_event = cuda.Event()
                        end_event = cuda.Event()
                        
                        start_event.record()
                        
                        # Launch kernel using current context's function handle
                        self.kernel(
                            A_data_gpu, A_indices_gpu, A_indptr_gpu,
                            B_data_gpu, B_indices_gpu, B_indptr_gpu,
                            C_gpu, np.int32(sub_adj.shape[0]),
                            np.int32(sub_adj.shape[1]),
                            np.int32(sub_feat.shape[1]),
                            block=self.block,
                            grid=grid
                        )
                        
                        end_event.record()
                        end_event.synchronize()
                        kernel_time = start_event.time_till(end_event)

                        # Get result in current context
                        result = np.empty((sub_adj.shape[0], sub_feat.shape[1]), dtype=np.float32)
                        cuda.memcpy_dtoh(result, C_gpu)
                        
                        # Include timing in results tuple
                        results.append((idx, result[:len(nodes_idx)], nodes_idx, kernel_time))

                    finally:
                        # Clean up GPU memory in current context
                        for ptr in gpu_ptrs:
                            ptr.free()

                except cuda.Error as e:
                    print(f"CUDA error in batch item {idx}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error in process_batch: {e}")
            return []
        finally:
            if self.ctx:
                self.ctx.pop()
                self.ctx = None
                self.kernel = None
                self.mod = None

    def process_clusters(self, cluster_data):
        """Process clusters with improved error handling"""
        all_results = {}
        
        # Process in batches without threading
        batches = [
            list(enumerate(cluster_data[i:i + self.batch_size]))
            for i in range(0, len(cluster_data), self.batch_size)
        ]
        
        for batch in batches:
            try:
                batch_results = self.process_batch(batch)
                if batch_results:  # Only process if we got results
                    for idx, result, nodes_idx, timing in batch_results:
                        all_results[idx] = (result, nodes_idx, timing)  # Store timing
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
                
        return all_results

    def cleanup(self):
        """Enhanced cleanup"""
        self.free_gpu_memory(*self.allocated_memory.keys())
        self.allocated_memory.clear()
        if self.ctx:
            try:
                self.ctx.pop()
            except:
                pass
            self.ctx = None

if __name__ == '__main__':
    cuda.init()
    try:
        # Initialize kernel manager first
        kernel_manager = CUDAKernelManager.get_instance()
        try:
            kernel_manager.init_context()
        except Exception as e:
            print(f"Failed to initialize global kernel manager: {e}")
            raise

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
                avg_cluster_size = max(int(np.sqrt(num_nodes)), batch_size)  # Increased minimum size
                num_clusters = max(2, num_nodes // avg_cluster_size)  # Added upper limit
                
                print(f"Decomposing graph into {num_clusters} clusters")
                try:
                    # Try GPU-based clustering first
                    clusters = create_clusters_metis_bfs_gpu(adjacency_matrix, num_clusters)
                except Exception as e:
                    print(f"GPU clustering failed: {e}")
                    print("Falling back to CPU-based spectral clustering")

                
                previous_time = time.perf_counter()
                print(f"Decomposition Time elapsed: { previous_time - decomp_start:.4f} seconds")

                
                if not clusters:
                    raise RuntimeError("Failed to create clusters")
                    
                # Add size-based filtering
                clusters = [c for c in clusters if len(c) >= 2]  # Remove tiny clusters
                if not clusters:
                    # If all clusters were filtered out, create one cluster with all nodes
                    clusters = [list(range(num_nodes))]
                    
                print(f"Extracting {len(clusters)} clusters")
                # Prepare cluster data
                cluster_data = []
                for cluster_nodes in clusters:
                    sub_adj, sub_feat, nodes_idx, all_nodes = extract_submatrices(
                        adjacency_matrix, feature_matrix, cluster_nodes)
                    # Check if sub_adj and sub_feat are valid
                    if sub_adj is None or sub_feat is None or \
                    sub_adj.shape[0] == 0 or sub_adj.shape[1] == 0 or \
                    sub_feat.shape[0] == 0 or sub_feat.shape[1] == 0:
                        print(f"Skipping cluster due to empty submatrices.")
                        continue
                    cluster_data.append((sub_adj, sub_feat, nodes_idx, all_nodes))

                
                print(f"Cluster Extraction Time elapsed: { time.perf_counter() - previous_time:.4f} seconds")
                previous_time = time.perf_counter()
                
                decomp_time = time.perf_counter() - decomp_start

                # Start timing just the multiplication phase
                start_event = cuda.Event()
                end_event = cuda.Event()
                start_event.record()

                start_time = time.perf_counter()
                
                pipeline = GPUPipeline(batch_size=4)
                pipeline.init_kernel_manager()
                result_dict = pipeline.process_clusters(cluster_data)

                # display time elapsed
                previous_time = time.perf_counter()
                print(f"PipelineTime elapsed: { previous_time - start_time:.4f} seconds")
                
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

                
                print(f"Merge Time elapsed: { time.perf_counter() - previous_time:.4f} seconds")
                previous_time = time.perf_counter()

                end_event.record()
                end_event.synchronize()
                mult_time = start_event.time_till(end_event)

                end_time = time.perf_counter() - start_time

                start_time = time.perf_counter()

                # Verify the result
                is_correct = verify_result(result, adjacency_matrix, feature_matrix)

                end_time_cpu = time.perf_counter() - start_time

                if not is_correct:
                    print(f"Graph {name} failed verification.")
                else:
                    # Stop memory tracking and get results
                    stop_event.set()
                    
                    results.append({
                        "graph_index": index,
                        "graph_name": name,
                        "graph_type": graph_type,
                        "method": "pycuda_sparse_bfs_tensor_simplified_02",
                        "decomposition_time": decomp_time,
                        "multiplication_time": end_time,
                        "num_clusters": len(clusters),
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
                        "method": "pycuda_sparse_bfs_tensor_simplified_02_total_time",
                        "decomposition_time": 0,  # Set to 0 since we're using total time
                        "multiplication_time": decomp_time + end_time, # Add decomp_time and mult_time
                        "num_clusters": len(clusters),
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
                        "method": "verify_time_cpu_sparse",
                        "decomposition_time": 0,  # Set to 0 since we're using total time
                        "multiplication_time": end_time_cpu, # Add decomp_time and mult_time
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
                if pipeline:
                    pipeline.cleanup()
                
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if kernel_manager and kernel_manager.context:
            try:
                kernel_manager.cleanup()
            except:
                pass

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