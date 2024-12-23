
import numpy as np
import scipy.io
import os
import json
from google.colab import drive
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# For running in Colab with T4 GPU with Tensor Cores
drive.mount('/content/drive')


matrix_dir = "/content/drive/MyDrive/GPU/matrices" 
metadata_file = "/content/drive/MyDrive/GPU/matrices/ss_matrix_metadata.json"  
result_file = "/content/drive/MyDrive/GPU/results/result_tensor_core_tf_sparse.json"  


def load_sparse_matrix(mtx_path):
    sparse_matrix = scipy.io.mmread(mtx_path).astype(np.float32)
    if sparse_matrix.shape[0] != sparse_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square. Found shape: {sparse_matrix.shape}")
    return sparse_matrix

# Generate a synthetic dense feature matrix (B) of size (#columns of A, 1000)
def generate_dense_feature_matrix(num_cols, num_features=1000):
    dense_matrix = np.random.rand(num_cols, num_features).astype(np.float16)  # FP16 for Tensor Core compatibility
    return dense_matrix

def tensor_core_matrix_multiplication(A, B):
    A_dense = tf.convert_to_tensor(A.toarray(), dtype=tf.float16)  
    B_dense = tf.convert_to_tensor(B, dtype=tf.float16)  

    start_time = time.time()
    C = tf.matmul(A_dense, B_dense)  # Tensor Core-accelerated multiplication
    elapsed_time = time.time() - start_time

    return elapsed_time


def load_metadata(metadata_file):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    with open(metadata_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":

    metadata = load_metadata(metadata_file)
    metadata_dict = {entry["name"]: entry for entry in metadata}  # Map metadata by matrix name


    mtx_files = [f for f in os.listdir(matrix_dir) if f.endswith('.mtx')]
    total_files = len(mtx_files)

    print(f"Found {total_files} .mtx files in the directory: {matrix_dir}")
    if not mtx_files:
        print(f"No .mtx files found in directory {matrix_dir}.")
    else:
        results = []
        processed_count = 0

        for mtx_file in mtx_files:
            mtx_path = os.path.join(matrix_dir, mtx_file)
            try:
                print(f"\nProcessing matrix file {processed_count + 1}/{total_files}: {mtx_file}")


                A = load_sparse_matrix(mtx_path)


                matrix_name = os.path.splitext(mtx_file)[0] 
                metadata_entry = metadata_dict.get(matrix_name, {})
                sparsity = metadata_entry.get("sparsity", None)

                if sparsity is None:
                    print(f"Sparsity information not found for {matrix_name}. Skipping.")
                    continue


                B = generate_dense_feature_matrix(A.shape[1])

                multiply_time = tensor_core_matrix_multiplication(A, B)


                results.append({
                    "matrix_name": mtx_file,
                    "num_rows": A.shape[0],
                    "num_cols": A.shape[1],
                    "sparsity": sparsity,
                    "multiply_time_seconds": multiply_time
                })

                print(f"Matrix multiplication for {mtx_file} completed in {multiply_time:.4f} seconds.")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {mtx_file}: {e}")


        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nProcessing completed. Total matrices processed: {processed_count}/{total_files}.")
        print(f"Results saved to: {result_file}")
