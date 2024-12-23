import os
import subprocess
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# Function to find MSVC compiler path dynamically
def find_msvc_path():
    try:
        vswhere_path = r"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
        output = subprocess.check_output(
            [
                vswhere_path,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                "VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64",
            ],
            text=True,
        ).strip()
        return output
    except subprocess.CalledProcessError:
        raise RuntimeError("MSVC compiler path not found")


# Set CUDA compiler path
os.environ["CUDA_PATH"] = str(Path(subprocess.check_output(["where", "nvcc"], text=True).splitlines()[0]).parent.parent)

# Set MSVC compiler path
os.environ["PATH"] = find_msvc_path() + os.pathsep + os.environ["PATH"]


def load_gpu_kernel(mod: str, *func: str) -> cuda.Function:
    kernels = [path.stem for path in Path("kernels").glob("*.cu")]
    if mod not in kernels:
        raise ValueError(f"Module {mod} not found in kernels")

    with open(Path("kernels") / f"{mod}.cu", "r") as f:
        kernel_source = f.read()

    # Compile kernel
    mod = SourceModule(kernel_source, no_extern_c=True)
    return (mod.get_function(f) for f in func)


def allocate_gpu_memory(data: np.ndarray) -> cuda.DeviceAllocation:
    """Allocate memory on the GPU and copy data to it."""
    try:
        gpu_mem = cuda.mem_alloc(data.nbytes)
    except cuda.MemoryError:
        raise RuntimeError("Out of memory on the GPU")
    cuda.memcpy_htod(gpu_mem, data)
    return gpu_mem


def fetch_gpu_data(gpu_mem: cuda.DeviceAllocation, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Fetch data from GPU memory back to the host."""
    host_data = np.empty(shape, dtype=dtype)
    cuda.memcpy_dtoh(host_data, gpu_mem)
    return host_data