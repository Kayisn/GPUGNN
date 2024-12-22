import os
import subprocess
from pathlib import Path


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
