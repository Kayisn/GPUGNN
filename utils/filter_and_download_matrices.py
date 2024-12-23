import json
import os
import shutil
from pathlib import Path

import ssgetpy


def filter_matrices(filename, row_range=None, col_range=None, sparsity_range=None):
    """
    Filter matrices based on the row, column, and sparsity range.

    Parameters:
        filename (str): Path to the metadata file.
        row_range (tuple): Min and max number of rows.
        col_range (tuple): Min and max number of columns.
        sparsity_range (tuple): Min and max sparsity.

    Returns:
        list: Filtered matrix metadata.
    """
    if not os.path.exists(filename):
        print(f"Metadata file {filename} not found. Maybe run the fetch_and_store_results() first!")
        return []

    with open(filename, "r") as f:
        matrix_metadata = json.load(f)

    filtered = []
    for matrix in matrix_metadata:
        if row_range and not (row_range[0] <= matrix["rows"] <= row_range[1]):
            continue
        if col_range and not (col_range[0] <= matrix["cols"] <= col_range[1]):
            continue
        if sparsity_range and not (sparsity_range[0] <= matrix["sparsity"] <= sparsity_range[1]):
            continue
        filtered.append(matrix)

    return filtered


def download_and_save_matrices(filtered_matrices, dest_dir: Path, max_matrices=10):
    """
    Download filtered matrices and save their .mtx files directly to the destination directory, instead of under a sub-folder.

    Parameters:
        filtered_matrices (list): List of filtered matrix metadata.
        dest_dir (str): Path to the directory where the .mtx files should be saved.
        max_matrices (int): Maximum number of matrices to process.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for matrix_meta in filtered_matrices:
        if processed_count >= max_matrices:
            break

        matrix_name = matrix_meta["name"]
        mtx_file_path = dest_dir / f"{matrix_name}.mtx"

        # Check if the .mtx file already exists
        if mtx_file_path.exists():
            print(f"Matrix '{matrix_name}' already exists at {mtx_file_path}. Skipping download.")
            continue

        print(f"Downloading matrix: {matrix_name}")
        try:
            matrix = ssgetpy.search(name=matrix_name, limit=10)[0]  # Search for the matrix
            matrix.download(format="MM", destpath=dest_dir, extract=True)  # Download matrix

            # Find and move .mtx file
            for mtx_file in (dest_dir / matrix_name).glob("*.mtx"):
                shutil.move(mtx_file, mtx_file_path)

            # Remove the now-empty extracted folder
            shutil.rmtree(dest_dir / matrix_name)
            print(f"Matrix '{matrix_name}' saved as: {mtx_file_path}")
            processed_count += 1
        except Exception as e:
            print(f"Error downloading matrix '{matrix_name}': {e}")


if __name__ == "__main__":
    # Define filtering criteria
    row_range = (10000, 50000)
    col_range = (10000, 50000)
    sparsity_range = (0.9, 1.0)

    dest_dir = Path("graphs") / "ssmatrices"  # Directory to save the .mtx files
    metadata_file = Path("graphs") / "ssmatrices_metadata.json"  # Path to the metadata file

    # Filter matrices
    filtered_matrices = filter_matrices(
        filename=metadata_file, row_range=row_range, col_range=col_range, sparsity_range=sparsity_range
    )

    print(f"Found {len(filtered_matrices)} matrices matching the criteria")

    download_and_save_matrices(filtered_matrices, dest_dir, max_matrices=10)
