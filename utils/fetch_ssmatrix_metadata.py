import os
import json
import scipy.io
from pathlib import Path

def load_metadata(filename):
    """
    Load the metadata file.

    Parameters:
        filename (str): Path to the metadata file.

    Returns:
        dict: Metadata for the matrices.
    """
    if not os.path.exists(filename):
        print(f"Metadata file {filename} not found.")
        return {}

    with open(filename, "r") as f:
        return {matrix["name"]: matrix for matrix in json.load(f)}

def analyze_matrices(matrix_dir, metadata_file):
    """
    Analyze all .mtx matrices in the directory and display their properties.

    Parameters:
        matrix_dir (str): Path to the directory containing .mtx files.
        metadata_file (str): Path to the metadata file for additional matrix info.
    """
    # Load metadata
    metadata = load_metadata(metadata_file)

    # Iterate over all .mtx files in the directory
    for mtx_file in Path(matrix_dir).glob("*.mtx"):
        matrix_name = mtx_file.stem

        try:
            # Load the matrix
            matrix = scipy.io.mmread(mtx_file)
            rows, cols = matrix.shape
            nnz = matrix.nnz

            sparsity = metadata.get(matrix_name, {}).get("sparsity", 1 - (nnz / (rows * cols)))

            group = metadata.get(matrix_name, {}).get("group", "Unknown")
            kind = metadata.get(matrix_name, {}).get("kind", "Unknown")

            print(f"Matrix Name: {matrix_name}")
            print(f"  Group: {group}")
            print(f"  Kind: {kind}")
            print(f"  Rows: {rows}")
            print(f"  Columns: {cols}")
            print(f"  Non-Zero Elements: {nnz}")
            print(f"  Sparsity: {sparsity:.6f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error processing matrix {matrix_name}: {e}")

if __name__ == "__main__":
    matrix_dir = "../ssmatrices"  # Directory containing .mtx files
    metadata_file = "../ssmatrices/ss_matrix_metadata.json"  # Metadata file

    analyze_matrices(matrix_dir, metadata_file)
