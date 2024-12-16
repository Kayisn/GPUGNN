import numpy as np
import scipy.sparse as sp

def verify_result(gpu_result, adj_matrix, feature_matrix, rtol=1e-3, atol=1e-4):
    """
    Verify GPU computation result against CPU reference calculation for GNN operations
    
    Args:
        gpu_result: numpy array from GPU calculation
        adj_matrix: Graph adjacency matrix (can be any scipy sparse format)
        feature_matrix: Node feature matrix (can be any scipy sparse format)
        rtol, atol: Relative and absolute tolerance for comparison
    
    Returns:
        bool: True if result matches CPU calculation within tolerance
    """
    # Convert inputs to CSR format if they aren't already
    if not isinstance(adj_matrix, sp.csr_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)
    if not isinstance(feature_matrix, sp.csr_matrix):
        feature_matrix = sp.csr_matrix(feature_matrix)
    
    # Verify input shapes are compatible
    if adj_matrix.shape[1] != feature_matrix.shape[0]:
        print(f"Error: Incompatible shapes for GNN operation")
        print(f"Adjacency matrix: {adj_matrix.shape}")
        print(f"Feature matrix: {feature_matrix.shape}")
        return False
        
    # Calculate reference result on CPU
    cpu_result = (adj_matrix @ feature_matrix).toarray()
    
    # Check if shapes match
    if not isinstance(gpu_result, np.ndarray):
        gpu_result = gpu_result.toarray()
    if cpu_result.shape != gpu_result.shape:
        print(f"Warning: Shape mismatch - CPU: {cpu_result.shape}, GPU: {gpu_result.shape}")
        return False
    
    # Compare results
    is_close = np.allclose(cpu_result, gpu_result, rtol=rtol, atol=atol)
    if not is_close:
        max_diff = np.max(np.abs(cpu_result - gpu_result))
        relative_diff = np.max(np.abs((cpu_result - gpu_result) / (cpu_result + 1e-10)))
        print(f"Warning: Results differ!")
        print(f"Max absolute difference: {max_diff}")
        print(f"Max relative difference: {relative_diff}")
        
        # Additional debugging info
        print(f"CPU result - min: {np.min(cpu_result)}, max: {np.max(cpu_result)}, mean: {np.mean(cpu_result)}")
        print(f"GPU result - min: {np.min(gpu_result)}, max: {np.max(gpu_result)}, mean: {np.mean(gpu_result)}")
        
        # Show difference distribution
        diff = np.abs(cpu_result - gpu_result)
        print(f"Difference distribution:")
        print(f"- 25th percentile: {np.percentile(diff, 25)}")
        print(f"- 50th percentile: {np.percentile(diff, 50)}")
        print(f"- 75th percentile: {np.percentile(diff, 75)}")
        print(f"- 99th percentile: {np.percentile(diff, 99)}")
    
    return is_close