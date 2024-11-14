# GPUGNN - GPU-Accelerated Graph Neural Network Operations
For a GPU computing class at UVIC

## Project Overview
This project implements and evaluates different GPU-accelerated approaches for sparse matrix operations commonly found in Graph Neural Networks (GNNs), with a focus on sparse matrix multiplication.


## Current State
- Implemented multiple CUDA kernel variants for sparse matrix multiplication:
  - Basic sparse implementation
  - CSR-CSC optimized version
  - Tiled version with shared memory
  - Coalesced memory access version
  - bfs based decomposition
- Support for synthetic (Erdős-Rényi, Scale-Free) and real-world graphs (real-world are still test graphs)
- Performance profiling and metrics collection
- Thread-safe result logging and verification
- Memory usage tracking and occupancy analysis

## Limitations
1. Memory Constraints:
   - Limited by available GPU memory for large graphs
   - No out-of-core processing support

2. Performance:
   - Current implementations may not handle highly skewed degree distributions optimally
   - Limited load balancing for irregular workloads
   - No multi-GPU support

3. Features:
   - Limited to single-precision floating point
   - No support for dynamic graphs
   - Limited graph formats supported (CSR/CSC only)

## Future Objectives
1. Short-term:
   - Implement load balancing schemes for irregular graphs
   - Optimize memory access patterns further
   - Add support for blocked sparse formats

2. Medium-term:
   - Add multi-GPU support
   - Implement out-of-core processing for large graphs
   - Implement autotuning for kernel parameters

3. Long-term:
   - Develop hybrid CPU-GPU processing strategies
   - Add distributed computing support
   - Implement specialized kernels for different graph types
   - Add support for graph compression techniques

## Requirements
- CUDA Toolkit 12.6
- Python 3.8+
- Dependencies:
  - PyCUDA
  - NumPy
  - SciPy
  - NetworkX
  - FileLock

## Usage
```bash

# Generate test graphs
python graph_generation.py

# Run all or a subset of methods on all generated graphs
# Edit the file to run the methods you're interested in
python script_process_graphs.py