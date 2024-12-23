#!/usr/bin/env python3

import json
import pickle
import time
import numpy as np
import scipy.sparse as sp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from verification import verify_result

# ================================
# Load your test graphs
# ================================
with open("gnn_test_graphs_with_features.pkl","rb") as f:
    graphs = pickle.load(f)

# ================================
# Multi-layer BFS kernel
# ================================
MULTI_LAYER_BFS_KERNEL = r'''
__global__ void multi_layer_bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*       __restrict__ distances, 
    const int  num_nodes,
    const int  d,   // BFS depth limit
    // Frontier
    const int* __restrict__ current_frontier,
    const int  current_frontier_size,
    // Next frontier
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size
){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid >= current_frontier_size) return;

    int node = current_frontier[tid];
    int dist_node = distances[node];
    if(dist_node >= d){
        return;
    }

    int start = row_ptr[node];
    int end   = row_ptr[node+1];
    for(int e=start; e<end; e++){
        int nbr = col_idx[e];
        if(nbr<0 || nbr>=num_nodes) continue;

        // if unvisited => set dist=dist_node+1
        if(atomicCAS(&distances[nbr], -1, dist_node+1)==-1){
            // push
            int idx= atomicAdd(next_frontier_size,1);
            next_frontier[idx]= nbr;
        }
    }
}
'''.strip()

def multi_layer_bfs_partition(adjacency, d=2, seeds_per_batch=4, block_size=128):
    """
    Multi-layer BFS Partition:
      - BFS expansions up to distance d
      - seeds in batches
      - merges single-node clusters if needed
      - ghost expansions
    Return: list of clusters (each is primary + ghost).
    """
    # Note: remove extern "C" from the kernel code and keep no_extern_c=True.
    module = SourceModule(MULTI_LAYER_BFS_KERNEL, no_extern_c=True)
    bfs_kernel = module.get_function("multi_layer_bfs_kernel")

    if not sp.isspmatrix_csr(adjacency):
        adjacency = adjacency.tocsr()
    n = adjacency.shape[0]
    row_ptr = adjacency.indptr.astype(np.int32)
    col_idx = adjacency.indices.astype(np.int32)

    # device mem
    d_row = cuda.mem_alloc(row_ptr.nbytes)
    d_col = cuda.mem_alloc(col_idx.nbytes)
    cuda.memcpy_htod(d_row, row_ptr)
    cuda.memcpy_htod(d_col, col_idx)

    # We'll store cluster assignment on host
    cluster_assign = np.full(n, -1, dtype=np.int32)
    distances      = np.full(n, -1, dtype=np.int32)  # BFS expansions
    d_dist         = cuda.mem_alloc(distances.nbytes)

    d_front_curr = cuda.mem_alloc(n*4)
    d_front_next = cuda.mem_alloc(n*4)
    d_size_curr  = cuda.mem_alloc(4)
    d_size_next  = cuda.mem_alloc(4)

    degrees   = np.diff(adjacency.indptr)
    uncovered = set(range(n))
    cluster_id= 0

    def run_seed_bfs(seed):
        # BFS up to distance d
        distances.fill(-1)
        distances[seed] = 0
        cuda.memcpy_htod(d_dist, distances)

        front_arr = np.array([seed], dtype=np.int32)
        size = np.int32(1)
        cuda.memcpy_htod(d_front_curr, front_arr)
        cuda.memcpy_htod(d_size_curr, size)

        while True:
            next_size_h = np.zeros(1, dtype=np.int32)
            cuda.memcpy_htod(d_size_next, next_size_h)

            front_size_host = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(front_size_host, d_size_curr)
            fs = int(front_size_host[0])
            if fs == 0:
                break

            threads = min(block_size, fs)
            blocks  = ((fs + threads - 1) // threads, 1, 1)

            bfs_kernel(
                d_row, d_col,
                d_dist,
                np.int32(n),
                np.int32(d),
                d_front_curr,
                np.int32(fs),
                d_front_next,
                d_size_next,
                block=(threads,1,1),
                grid=blocks
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(next_size_h, d_size_next)
            fs_next = int(next_size_h[0])
            if fs_next == 0:
                break

            cuda.memcpy_dtod(d_front_curr, d_front_next, fs_next*4)
            cuda.memcpy_htod(d_size_curr, np.int32(fs_next))

        # BFS done => distances => assigned
        cuda.memcpy_dtoh(distances, d_dist)

    # BFS expansions
    while uncovered:
        deg_list = [(degrees[x], x) for x in uncovered]
        deg_list.sort(reverse=True, key=lambda x: x[0])
        pick = deg_list[:seeds_per_batch]
        if not pick:
            break
        for _, seed in pick:
            if seed not in uncovered:
                continue
            run_seed_bfs(seed)
            assigned_nodes = np.where(distances >= 0)[0]
            for nd in assigned_nodes:
                if nd in uncovered:
                    cluster_assign[nd] = cluster_id
                    uncovered.remove(nd)
            cluster_id += 1

    # Build primary clusters
    from collections import defaultdict
    c_map = defaultdict(list)
    for nd, cid in enumerate(cluster_assign):
        if cid >= 0:
            c_map[cid].append(nd)
    clusters_primary = [v for k,v in sorted(c_map.items()) if len(v) > 0]

    # ghost expansions
    rowp = adjacency.indptr
    colx = adjacency.indices
    assignment_dict = {}
    for cid, c_nodes in enumerate(clusters_primary):
        for nd in c_nodes:
            assignment_dict[nd] = cid

    ghosted_clusters = []
    for cid, c_nodes in enumerate(clusters_primary):
        cset = set(c_nodes)
        for nd in c_nodes:
            st = rowp[nd]
            en = rowp[nd+1]
            for e in range(st,en):
                nbr = colx[e]
                if assignment_dict.get(nbr, -999) != cid:
                    cset.add(nbr)
        ghosted_clusters.append(sorted(cset))

    # free BFS device mem
    d_row.free(); d_col.free()
    d_dist.free()
    d_front_curr.free()
    d_front_next.free()
    d_size_curr.free()
    d_size_next.free()

    return ghosted_clusters

# ================================
# 2) Hybrid spMM: tile-dense + CSR
# ================================
TILE_DENSE_SRC = r'''
#define BLOCK 16
__global__ void tile_dense_mm(
    const float* A,
    const float* B,
    float*       C,
    int N,int K,int M
){
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    int row= blockIdx.y*BLOCK + threadIdx.y;
    int col= blockIdx.x*BLOCK + threadIdx.x;

    float val=0.0f;
    for(int t=0; t< ( (K+BLOCK-1)/BLOCK ); t++){
        int Acol = t*BLOCK + threadIdx.x;
        int Arow = row;

        int Brow= t*BLOCK + threadIdx.y;
        int Bcol= col;

        if(Arow<N && Acol<K){
            As[threadIdx.y][threadIdx.x] = A[Arow*K + Acol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(Brow<K && Bcol<M){
            Bs[threadIdx.y][threadIdx.x] = B[Brow*M + Bcol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for(int i=0; i<BLOCK; i++){
            val += As[threadIdx.y][i]*Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if(row<N && col<M){
        C[row*M + col] = val;
    }
}
'''.strip()

CSR_SPMM_SRC = r'''
__global__ void csr_spmm(
    const float* __restrict__ A_data,
    const int*   __restrict__ A_indices,
    const int*   __restrict__ A_indptr,
    const float* __restrict__ B_data,
    const int*   __restrict__ B_indices,
    const int*   __restrict__ B_indptr,
    float*       __restrict__ C,
    int N,int K,int M
){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < N && col < M){
        float sum = 0.0f;
        int start = A_indptr[row];
        int end   = A_indptr[row+1];
        for(int e=start; e<end; e++){
            int kk    = A_indices[e];
            float a_v = A_data[e];
            int bs    = B_indptr[kk];
            int be    = B_indptr[kk+1];

            // binary search over B row
            int left = bs, right = be-1;
            while(left <= right){
                int mid = (left+right) >> 1;
                int bc  = B_indices[mid];
                if(bc == col){
                    sum += a_v * B_data[mid];
                    break;
                }
                if(bc < col) left=mid+1; 
                else         right=mid-1;
            }
        }
        C[row*M + col] = sum;
    }
}
'''.strip()

class HybridSpMM:
    def __init__(self):
        # Remove extern "C" from the source strings but keep no_extern_c=True
        self.ctx       = cuda.Device(0).make_context()
        self.mod_dense = SourceModule(TILE_DENSE_SRC, no_extern_c=True)
        self.mod_csr   = SourceModule(CSR_SPMM_SRC,   no_extern_c=True)

        self.kernel_dense = self.mod_dense.get_function("tile_dense_mm")
        self.kernel_csr   = self.mod_csr.get_function("csr_spmm")

    def close(self):
        if self.ctx:
            self.ctx.pop()
            self.ctx = None

    def multiply_submatrix(self, sub_adj, sub_feat):
        """
        sub_adj: NxK, sub_feat: KxF
        if small & dense => tile-dense-mm
        else => csr_spmm
        pinned mem in main thread => no invalid context
        """
        N,K  = sub_adj.shape
        K2,F = sub_feat.shape
        assert K == K2

        nnz       = sub_adj.nnz
        total_els = N*K
        density   = nnz / (total_els + 1e-9)

        # simple threshold
        if total_els < 250_000 and density > 0.25:
            # Use tile-dense-mm
            A_arr = sub_adj.toarray().astype(np.float32)
            B_arr = sub_feat.toarray().astype(np.float32)
            C_arr = np.zeros((N,F), dtype=np.float32)

            # pinned
            A_pin = cuda.register_host_memory(A_arr)
            B_pin = cuda.register_host_memory(B_arr)
            C_pin = cuda.register_host_memory(C_arr)

            dA = cuda.mem_alloc(A_arr.nbytes)
            dB = cuda.mem_alloc(B_arr.nbytes)
            dC = cuda.mem_alloc(C_arr.nbytes)

            cuda.memcpy_htod(dA, A_pin)
            cuda.memcpy_htod(dB, B_pin)

            block = (16,16,1)
            grid  = ((F + 15)//16, (N + 15)//16, 1)
            self.kernel_dense(
                dA, dB, dC,
                np.int32(N), np.int32(K), np.int32(F),
                block=block, grid=grid
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(C_pin, dC)
            dA.free(); dB.free(); dC.free()
            return C_arr
        else:
            # Use csr_spmm
            dataA  = sub_adj.data.astype(np.float32)
            idxA   = sub_adj.indices.astype(np.int32)
            ptrA   = sub_adj.indptr.astype(np.int32)
            dataB  = sub_feat.data.astype(np.float32)
            idxB   = sub_feat.indices.astype(np.int32)
            ptrB   = sub_feat.indptr.astype(np.int32)

            dA_data = cuda.mem_alloc(dataA.nbytes)
            dA_idx  = cuda.mem_alloc(idxA.nbytes)
            dA_ptr  = cuda.mem_alloc(ptrA.nbytes)
            dB_data = cuda.mem_alloc(dataB.nbytes)
            dB_idx  = cuda.mem_alloc(idxB.nbytes)
            dB_ptr  = cuda.mem_alloc(ptrB.nbytes)

            C_out = np.zeros((N,F),dtype=np.float32)
            dC    = cuda.mem_alloc(C_out.nbytes)

            # pinned
            dataA_pin = cuda.register_host_memory(dataA)
            idxA_pin  = cuda.register_host_memory(idxA)
            ptrA_pin  = cuda.register_host_memory(ptrA)
            dataB_pin = cuda.register_host_memory(dataB)
            idxB_pin  = cuda.register_host_memory(idxB)
            ptrB_pin  = cuda.register_host_memory(ptrB)
            C_out_pin = cuda.register_host_memory(C_out)

            # copy
            cuda.memcpy_htod(dA_data, dataA_pin)
            cuda.memcpy_htod(dA_idx,  idxA_pin)
            cuda.memcpy_htod(dA_ptr,  ptrA_pin)
            cuda.memcpy_htod(dB_data, dataB_pin)
            cuda.memcpy_htod(dB_idx,  idxB_pin)
            cuda.memcpy_htod(dB_ptr,  ptrB_pin)

            block = (32,32,1)
            grid  = ((F + 31)//32, (N + 31)//32, 1)

            self.kernel_csr(
                dA_data, dA_idx, dA_ptr,
                dB_data, dB_idx, dB_ptr,
                dC,
                np.int32(N), np.int32(K), np.int32(F),
                block=block, grid=grid
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(C_out_pin, dC)
            dA_data.free(); dA_idx.free(); dA_ptr.free()
            dB_data.free(); dB_idx.free(); dB_ptr.free()
            dC.free()
            return C_out

# ================================
# Main
# ================================
def main():
    final_results = []
    spmm = None
    try:
        spmm = HybridSpMM()  # single GPU context in main thread

        for graph_info in graphs:
            idx  = graph_info["index"]
            name = graph_info["name"]
            n    = graph_info["num_nodes"]
            s    = graph_info["sparsity"]
            graph= graph_info["graph"]
            fm   = graph_info["feature_matrix"]

            print(f"\n--- Graph {idx}: {name}, n={n}, s={s} ---")

            adjacency = sp.lil_matrix((n,n),dtype=np.float32)
            for node in graph.nodes:
                for nbr in graph.neighbors(node):
                    adjacency[node, nbr] = 1.0
            adjacency = adjacency.tocsr()
            feature   = sp.csr_matrix(fm, dtype=np.float32)

            # 1) BFS partition
            t0 = time.time()
            clusters = multi_layer_bfs_partition(adjacency, d=2, seeds_per_batch=4, block_size=128)
            t1 = time.time()
            bfs_time = t1 - t0
            print(f"Multi-layer BFS => {len(clusters)} clusters, took {bfs_time:.4f}s")

            # 2) Extraction
            def extract_submat(c):
                cset      = set(c)
                neighbors = set()
                for nd in c:
                    st = adjacency.indptr[nd]
                    en = adjacency.indptr[nd+1]
                    neighbors.update(adjacency.indices[st:en])
                all_nodes = sorted(cset | neighbors)
                sub_adj   = adjacency[list(c), :][:, all_nodes].tocsr()
                sub_feat  = feature[all_nodes, :]
                return sub_adj, sub_feat, c

            submatrices = []
            t2 = time.time()
            for c in clusters:
                if len(c) < 1:
                    continue
                sA, sF, idxs = extract_submat(c)
                if sA.nnz > 0:
                    submatrices.append((sA, sF, idxs))
            t3 = time.time()
            extract_time = t3 - t2
            print(f"Extraction: {len(submatrices)} subgraphs => {extract_time:.4f}s")

            # 3) spMM
            t4 = time.time()
            spmm_results = []
            for (subA, subF, node_list) in submatrices:
                outC = spmm.multiply_submatrix(subA, subF)
                spmm_results.append((outC,node_list))
            t5 = time.time()
            spmm_time = t5 - t4
            print(f"spMM time = {spmm_time:.4f}s")

            # 4) Merge
            finalC = np.zeros((n, feature.shape[1]), dtype=np.float32)
            count  = np.zeros(n, dtype=np.int32)

            t6 = time.time()
            for (matrix, cidx) in spmm_results:
                c = list(cidx)
                finalC[c] += matrix[:len(c)]
                count[c]  += 1
            mask = (count != 0)
            finalC[mask] /= count[mask,None]
            t7 = time.time()
            merge_time = t7 - t6
            print(f"Merging results => {merge_time:.4f}s")

            # 5) Verify
            t8 = time.time()
            is_ok = verify_result(finalC, adjacency, feature)
            t9 = time.time()
            verify_time = t9 - t8
            print(f"Verify => {is_ok}, took {verify_time:.4f}s")

            if is_ok:
                final_results.append({
                    "graph_index": idx,
                    "graph_name": name,
                    "graph_type:" : graph_info["type"],
                    "num_clusters": len(clusters),
                    "decomp": bfs_time,
                    "extract": extract_time,
                    "mult": spmm_time,
                    "merge": merge_time,
                    "verify_time": verify_time,
                    "is_correct": True,
                    "num_nodes": n,
                    "sparsity": s
                })

    finally:
        if spmm:
            spmm.close()

    # Save
    if final_results:
        with open("multi_layer_bfs_advanced_timing.json","w") as f:
            json.dump(final_results, f, indent=2)
        print("Saved results => multi_layer_bfs_advanced_timing.json.")


if __name__=="__main__":
    main()