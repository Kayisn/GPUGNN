import subprocess

# List of scripts to run
scripts = [
    #'chatgpt_cupy_sparse.py',
    'chatgpt_pytorch_sparse.py',
    'numpy_csr.py'
]
# I separate them because I have different environments for pycuda and comment out the other scripts to run only the pycuda script
scripts_pycuda = [
    #'chatgpt_pycuda_decomposed.py',
    #'chatgpt_pycuda_sparse.py',
    #'bfs_pycuda_sparse.py',
    #'bfs_pycuda_sparse_edgecount.py',
    #'bfs_pycuda_sparse_gptsimprovements.py',
    #'bfs_pycuda_batch.py',
    #'bfs_pycuda_sparse_piped_errorhandling.py',
    #'bfs_pycuda_edge_throughput.py',
    #'bfs_pycuda_edge_throughput_evolve.py',
    #'bfs_pycuda_edge_throughput_kernel_improve_1.py',
    #'bfs_pycuda_edge_throughput_piped.py',
    #'bfs_pycuda_edge_throughput_autodecomp.py',
    #'bfs_pycuda_sparse_piped_compressed.py',
    #'bfs_pycuda_tensor.py',
    #'bfs_pycuda_tensor_01.py',
    'bfs_pycuda_tensor_02.py',
    #'bfs_pycuda_tensor_01_simplified.py',
    #'bfs_pycuda_tensor_tiles.py',
    #'chatgpt_pycuda_sparse_coalesced.py',
    #'chatgpt_pycuda_sparse_snap.py',
    #'claude_pycuda_sparse.py',
    #'claude_pycuda_sparse_tiled.py',
    #'claude_pycuda_sparse_tiled_coalesced.py',
    #'claude_pycuda_sparse_tiled_instrumented.py',
    #claude_pycuda_sparse_csr_csc.py',
    #'chatgpt_pycuda_sparse_csr_csc.py',
]
scripts = scripts_pycuda

# Run each script sequentially
for script in scripts:
    try:
        print(f"Running {script}...")
        result = subprocess.run(['python', script], check=True)
        print(f"Completed {script} with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script}: {e}")

print("All scripts have been executed.")