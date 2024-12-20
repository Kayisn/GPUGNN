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
import networkx as nx

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

import pycuda.driver as cuda

def get_num_sms():
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)

import concurrent

from verification import verify_result

# Load graphs
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

import os

# Set CUDA compiler path before importing pycuda
os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\12.6'
os.environ['PATH'] = r'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\BuildTools\\VC\\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64' + os.pathsep + os.environ['PATH']



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



# Update CUDA kernel to use standard warp operations
EDGE_THROUGHPUT_KERNEL = """
__device__ __forceinline__ float atomicAdd_relaxed(float* address, float val) {
    return atomicAdd(address, val);
}

__global__ void edge_based_matmul(
    const float* __restrict__ edge_values,
    const int* __restrict__ edge_sources,
    const int* __restrict__ edge_targets,
    const float* __restrict__ feature_subset,
    float* __restrict__ output_matrix,
    const int batch_start,
    const int batch_size,
    const int feature_dim,
    const int num_nodes
) {
    // Calculate thread indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit if thread is outside bounds
    if (feature_idx >= feature_dim || tid >= batch_size) return;
    
    // Load edge data
    const int src = edge_sources[tid];
    const int dst = edge_targets[tid];
    const float edge_val = edge_values[tid];
    
    // Load and multiply feature value
    const float feat_val = feature_subset[src * feature_dim + feature_idx];
    const float result = edge_val * feat_val;
    
    // Accumulate result directly using atomic add
    atomicAdd_relaxed(&output_matrix[dst * feature_dim + feature_idx], result);
}
"""

def prepare_edge_lists(adjacency_matrix):
    """Convert adjacency matrix to edge lists for efficient GPU processing"""
    rows, cols = adjacency_matrix.nonzero()
    edge_values = adjacency_matrix.data.astype(np.float32)
    edge_sources = rows.astype(np.int32)
    edge_targets = cols.astype(np.int32)
    return edge_values, edge_sources, edge_targets

def prepare_feature_batch(edge_sources, feature_matrix):
    """Create optimized feature sub-matrix for a batch of edges"""
    # Get unique source nodes for this batch
    unique_sources = np.unique(edge_sources)
    
    # Create mapping from global to local indices
    source_to_local = {src: idx for idx, src in enumerate(unique_sources)}
    
    # Create local indices for the edge sources
    local_sources = np.array([source_to_local[src] for src in edge_sources], dtype=np.int32)
    
    # Extract only needed features
    if sp.issparse(feature_matrix):
        feature_subset = feature_matrix[unique_sources].toarray()
    else:
        feature_subset = feature_matrix[unique_sources]
        
    return feature_subset, local_sources, unique_sources

def sparse_matrix_multiply_edge_centric(adjacency_matrix, feature_matrix, batch_size=1024):
    """Edge-centric sparse matrix multiplication optimized for GPU throughput"""
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
    feature_matrix_gpu = cuda.mem_alloc(feature_matrix.nbytes)
    output_matrix_gpu = cuda.mem_alloc(num_nodes * num_features * 4)  # float32
    
    # Copy input data to GPU
    cuda.memcpy_htod(edge_values_gpu, edge_values)
    cuda.memcpy_htod(edge_sources_gpu, edge_sources)
    cuda.memcpy_htod(edge_targets_gpu, edge_targets)
    cuda.memcpy_htod(feature_matrix_gpu, feature_matrix.astype(np.float32))
    
    # Initialize output matrix to zeros
    cuda.memset_d32(output_matrix_gpu, 0, num_nodes * num_features)
    
    # Process edges in batches
    for batch_start in range(0, num_edges, batch_size):
        batch_end = min(batch_start + batch_size, num_edges)
        current_batch_size = batch_end - batch_start
        
        # Configure optimized thread and grid dimensions
        feature_dim = feature_matrix.shape[1]
        threads_per_block = (256, 1, 1)  # One thread per edge
        grid_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
        grid_y = (feature_dim + threads_per_block[1] - 1) // threads_per_block[1]
        grid = (grid_x, grid_y, 1)
        
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
            grid=grid
        )
    
    # Copy result back to host
    result = np.empty((num_nodes, num_features), dtype=np.float32)
    cuda.memcpy_dtoh(result, output_matrix_gpu)
    
    # Clean up
    edge_values_gpu.free()
    edge_sources_gpu.free()
    edge_targets_gpu.free()
    feature_matrix_gpu.free()
    output_matrix_gpu.free()
    
    return result

def decompose_edges(edge_values, edge_sources, edge_targets, max_edges_per_batch):
    """Decompose edges into batches while preserving locality and balancing workload by node degree"""
    num_edges = len(edge_values)
    
    # Calculate node degrees
    source_degrees = {}
    target_degrees = {}
    for src, dst in zip(edge_sources, edge_targets):
        source_degrees[src] = source_degrees.get(src, 0) + 1
        target_degrees[dst] = target_degrees.get(dst, 0) + 1
    
    # Create composite sort key: (source_degree, source_node, target_degree, target_node)
    edge_metadata = [(
        source_degrees[src], 
        src,
        target_degrees[dst],
        dst,
        i  # Keep original index
    ) for i, (src, dst) in enumerate(zip(edge_sources, edge_targets))]
    
    # Sort edges by degree and node ID
    sorted_edges = sorted(edge_metadata, key=lambda x: (-x[0], x[1], -x[2], x[3]))
    sorted_idx = [x[4] for x in sorted_edges]
    
    # Reorder edge data
    edge_values = edge_values[sorted_idx]
    edge_sources = edge_sources[sorted_idx]
    edge_targets = edge_targets[sorted_idx]
    
    # Create batches
    batches = []
    current_batch = ([], [], [])
    
    for i in range(num_edges):
        current_batch[0].append(edge_values[i])
        current_batch[1].append(edge_sources[i])
        current_batch[2].append(edge_targets[i])
        
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
    
    return batches

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

class CUDAContextManager:
    def __init__(self):
        self.context = None
        
    def __enter__(self):
        self.context = cuda.Device(0).make_context()
        return self
        
    def __exit__(self, *args):
        if self.context:
            self.context.pop()

class BatchIndexMapper:
    """Handles mapping between local and global indices for edge batches"""
    def __init__(self, edge_sources, edge_targets):
        # Get unique nodes
        self.unique_sources = np.unique(edge_sources)
        self.unique_targets = np.unique(edge_targets)
        
        # Create bidirectional mappings
        self.global_to_local_src = {src: idx for idx, src in enumerate(self.unique_sources)}
        self.local_to_global_src = {idx: src for src, idx in self.global_to_local_src.items()}
        
        # Convert edge indices to local space
        self.local_sources = np.array([self.global_to_local_src[src] for src in edge_sources], dtype=np.int32)
        
    def get_local_sources(self):
        return self.local_sources
    
    def get_feature_subset(self, feature_matrix):
        if sp.issparse(feature_matrix):
            return feature_matrix[self.unique_sources].toarray()
        return feature_matrix[self.unique_sources]
    
    def map_result_to_global(self, local_result, num_nodes, feature_dim):
        """Map results from local to global index space"""
        global_result = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for local_idx, global_idx in self.local_to_global_src.items():
            if local_idx < local_result.shape[0]:  # Safety check
                global_result[global_idx] = local_result[local_idx]
        return global_result

    def accumulate_result(self, global_result, batch_result, output_indices):
        """Accumulate batch results into the global result array using correct mapping"""
        for target_idx in self.unique_targets:
            if target_idx < global_result.shape[0]:  # Safety check
                global_result[target_idx] += batch_result[target_idx]
        return global_result

def process_edge_batch_pipelined(batch_queue, result_dict, lock, batch_size=1024):
    """Process edge batches with proper shape handling"""
    with CUDAContextManager():
        while True:
            try:
                idx, (edge_values, edge_sources, edge_targets, feature_matrix, num_nodes) = batch_queue.get_nowait()
                try:
                    # Create index mapper for this batch
                    index_mapper = BatchIndexMapper(edge_sources, edge_targets)
                    
                    # Get local indices and feature subset
                    local_sources = index_mapper.get_local_sources()
                    feature_subset = index_mapper.get_feature_subset(feature_matrix)
                    
                    # Create temporary sparse matrix using local indices
                    temp_matrix = sp.csr_matrix(
                        (edge_values, (local_sources, edge_targets)),
                        shape=(num_nodes, num_nodes)  # Use full shape here
                    )
                    
                    # Compute result maintaining full node dimensions
                    result = sparse_matrix_multiply_edge_centric(
                        temp_matrix, feature_subset, batch_size=batch_size)
                    
                    with lock:
                        result_dict[idx] = result
                except Exception as e:
                    print(f"Error processing batch {idx}: {e}")
                    print(f"Shapes - feature_subset: {feature_subset.shape}, result: {result.shape if 'result' in locals() else 'N/A'}")
                    with lock:
                        result_dict[idx] = None
            except Empty:
                break

class EdgePipeline:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.batch_queue = Queue()
        self.result_dict = {}
        self.lock = threading.Lock()
    
    def process_edges(self, edge_batches, feature_matrix, num_nodes, num_workers=2):
        # Convert feature matrix to dense if it's sparse
        if sp.issparse(feature_matrix):
            feature_matrix = feature_matrix.toarray()
            
        # Fill queue with batches
        for idx, batch in enumerate(edge_batches):
            edge_values, edge_sources, edge_targets = batch
            self.batch_queue.put((idx, (edge_values, edge_sources, edge_targets, feature_matrix, num_nodes)))
        
        # Create and start worker threads
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=process_edge_batch_pipelined,
                args=(self.batch_queue, self.result_dict, self.lock, self.batch_size)
            )
            worker.start()
            workers.append(worker)
        
        # Wait for all workers to complete
        for worker in workers:
            worker.join()
            
        # Collect and order results
        results = []
        for i in range(len(edge_batches)):
            if i in self.result_dict and self.result_dict[i] is not None:
                results.append(self.result_dict[i])
                
        return results

# Main execution code
if __name__ == "__main__":
    results = []
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

        # Setup memory monitoring
        free_mem, total_mem = cuda.mem_get_info()
        memory_idle = total_mem - free_mem
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        context = cuda.Device(0).make_context()

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
            
            # Calculate batch size dynamically based on GPU SMs
            num_sms = get_num_sms()
            threads_per_sm = 1024  # Adjust based on your GPU architecture
            total_threads = num_sms * threads_per_sm
            threads_per_edge = 1  # Define based on kernel requirements
            batch_size = total_threads // threads_per_edge

            if use_pipeline:
                max_edges_per_batch = min(batch_size, len(edge_values))
                edge_batches = decompose_edges(edge_values, edge_sources, edge_targets, max_edges_per_batch)
                decomp_time = time.perf_counter() - decomp_start

                # Process with pipeline
                with CUDAThreadManager.get_context():
                    start_event = cuda.Event()
                    end_event = cuda.Event()
                    start_event.record()
                    
                    pipeline = EdgePipeline(batch_size=max_edges_per_batch)
                    batch_results = pipeline.process_edges(
                        edge_batches, feature_matrix, num_nodes, num_workers=4)
                    
                    # Initialize result array
                    result = np.zeros((num_nodes, feature_matrix.shape[1]), dtype=np.float32)
                    
                    # Simply add results since target indices are already global
                    for batch_result in batch_results:
                        if batch_result is not None:
                            result += batch_result
                    
                    end_event.record()
                    end_event.synchronize()
                    mult_time = start_event.time_till(end_event)
            else:
                # Process entire graph at once
                decomp_time = time.perf_counter() - decomp_start
                
                with CUDAThreadManager.get_context():
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

            results.append({
                "graph_index": index,
                "graph_name": name,
                "graph_type": graph_type,
                "method": "pycuda_sparse_edge_throughput_evolve" + ("_piped" if use_pipeline else ""),
                "decomposition_time": decomp_time,
                "multiplication_time": mult_time / 1000.0,  # Convert ms to seconds
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
