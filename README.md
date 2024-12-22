# GPUGNN - GPU-Accelerated Graph Neural Network Operations
For a GPU computing class at UVIC

## Project Overview
This project implements and evaluates different GPU-accelerated approaches for sparse matrix operations commonly found in Graph Neural Networks (GNNs), with a focus on sparse matrix multiplication.

## Project Structure
GPUGNN/\
├── graphs/ - Contains generated test graphs\
├── plots/ - Contains generated performance plots\
├── reports/ - Contains performance profiling reports\
├── scripts/ - Contains Python scripts for each implementation\
├── utils/ - Contains utility scripts for graph generation and data processing\
├──── graph_generation.py - Graph generation script\
├──── verification.py - Graph verification script\
├── executer.py - Executes a method on a given set of graphs and optionally verifies the results\
├── gpugnn.py - Main script for running and profiling implementations\
├── results.json - Performance results database\
├── requirements.txt - Python dependencies
└── README.md - Project overview and usage instructions\

## Current State
- Implemented multiple CUDA kernel variants for sparse matrix multiplication:
  - Basic sparse implementation
  - CSR-CSC optimized version
  - Tiled version with shared memory
  - Coalesced memory access version
- Support for synthetic (Erdős-Rényi, Scale-Free) and real-world graphs (real-world are still test graphs)
- Performance profiling and metrics collection
- Result logging and verification
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
   - Add support for double precision
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
- CUDA Toolkit 11.0+
- Python 3.8+
- Dependencies:
  - PyCUDA
  - NumPy
  - SciPy
  - NetworkX

## Usage
```bash
# Generate test graphs
python utils/graph_generation.py

# Run basic sparse implementation
python gpugnn.py --methods chatgpt_pycuda_sparse --graphs 0

# Run chatgpt_pycuda_sparse and chatgpt_pytorch_dense with 10 warmup iterations and verify results
python gpugnn.py --warmup 10 --methods chatgpt_pycuda_sparse,chatgpt_pytorch_dense --verify

# Profile all PyCuda implementations on graph indeces 0, 1, and 2
python gpugnn.py --profile --methods pycuda --graphs 0-2

# Only profile the warmup iterations for the CuPy implementation on graph index 2
python gpugnn.py --profile --nvtx "warmup" --methods cupy --graphs 2

# Run all methods on all generated graphs with no warmup iterations
python gpugnn.py --profile --warmup 0 --methods all --graphs all
```