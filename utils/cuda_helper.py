from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import utils.init_compiler  # sets CUDA environment variables


def load_gpu_func(func_name):
    kernels = [path.stem for path in (Path("scripts") / "kernels").glob("*.cu")]
    if func_name not in kernels:
        raise ValueError(f"Function {func_name} not found in kernels")

    with open(Path("scripts") / "kernels" / f"{func_name}.cu", "r") as f:
        kernel_source = f.read()

    # Compile kernel
    mod = SourceModule(kernel_source)
    return mod.get_function(func_name)


def allocate_gpu_memory(data: np.ndarray) -> cuda.DeviceAllocation:
    """Allocate memory on the GPU and copy data to it."""
    gpu_mem = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(gpu_mem, data)
    return gpu_mem


def fetch_gpu_data(gpu_mem: cuda.DeviceAllocation, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Fetch data from GPU memory back to the host."""
    host_data = np.empty(shape, dtype=dtype)
    cuda.memcpy_dtoh(host_data, gpu_mem)
    return host_data
