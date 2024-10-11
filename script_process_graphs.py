import subprocess

# List of scripts to run
scripts = [
    'chatgpt_cupy_sparse.py',
    'chatgpt_pytorch_sparse.py',
    'chatgpt_pytorch_GCN.py'
]
# I separate them because I have different environments for pycuda and comment out the other scripts to run only the pycuda script
scripts_pycuda = [
    'chatgpt_pycuda.py'
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