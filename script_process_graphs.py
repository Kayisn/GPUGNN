import subprocess

# List of scripts to run
scripts = [
    'chatgpt_cupy_sparse.py',
    'chatgpt_pytorch_sparse.py',
    'sequential_numpy.py'
]
# I separate them because I have different environments for pycuda and comment out the other scripts to run only the pycuda script
scripts_pycuda = [
    #chatgpt_pycuda_decomposed.py',
    #'chatgpt_pycuda_sparse.py',
    #'claude_pycuda_sparse.py',
    'claude_pycuda_sparse_tiled.py',
    #'claude_pycuda_sparse_tiled_instrumented.py',
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