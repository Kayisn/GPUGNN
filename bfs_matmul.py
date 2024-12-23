#!/usr/bin/env python3

import json
import pickle
import time
import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import scipy.sparse as sp
from pycuda.compiler import SourceModule
from verification import verify_result

# -------------------------------------
# Load your test graphs
# -------------------------------------
with open("gnn_test_graphs_with_features.pkl", "rb") as f:
    graphs = pickle.load(f)

# =====================================
# Multi-seed BFS kernel
# =====================================
MULTI_SEED_BFS_KERNEL = r'''
extern "C" {

__global__ void multi_seed_bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ assignments,  // each node's primary cluster, or -1 if unassigned
    const int num_nodes,

    // frontier info
    const int* __restrict__ frontier_nodes,
    const int* __restrict__ frontier_clusters,
    const int frontier_size,

    // next frontier
    int* __restrict__ next_frontier_nodes,
    int* __restrict__ next_frontier_clusters,
    int* __restrict__ next_frontier_size
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int node    = frontier_nodes[tid];
    int cluster = frontier_clusters[tid];

    int start = row_ptr[node];
    int end   = row_ptr[node + 1];
    for (int e = start; e < end; e++){
        int nbr = col_idx[e];
        if (nbr < 0 || nbr >= num_nodes) continue;

        // If unassigned => claim
        if (atomicCAS(&assignments[nbr], -1, cluster) == -1){
            // Add neighbor to next frontier
            int idx = atomicAdd(next_frontier_size, 1);
            next_frontier_nodes[idx]    = nbr;
            next_frontier_clusters[idx] = cluster;
        }
    }
}

} // extern "C"
'''.strip()

def multi_seed_bfs_expansion(rowptr_gpu, colidx_gpu, assignments_gpu, num_nodes,
                             frontier_nodes_gpu, frontier_clusters_gpu, frontier_size_gpu,
                             next_frontier_nodes_gpu, next_frontier_clusters_gpu, next_frontier_size_gpu,
                             bfs_kernel, block_size):
    """
    One BFS expansion pass from the current frontier.
    """
    # read current frontier_size from device
    size_arr = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(size_arr, frontier_size_gpu)
    frontier_size = int(size_arr[0])
    if frontier_size == 0:
        return 0

    # reset next_frontier_size=0
    zero_val = np.int32(0)
    cuda.memcpy_htod(next_frontier_size_gpu, zero_val)

    # compute blocks
    blocks = max(1, (frontier_size + block_size -1)//block_size)

    # launch BFS
    bfs_kernel(
        rowptr_gpu,
        colidx_gpu,
        assignments_gpu,
        np.int32(num_nodes),
        frontier_nodes_gpu,
        frontier_clusters_gpu,
        np.int32(frontier_size),
        next_frontier_nodes_gpu,
        next_frontier_clusters_gpu,
        next_frontier_size_gpu,
        block=(block_size,1,1),
        grid=(blocks,1,1)
    )
    cuda.Context.synchronize()

    # read next_frontier_size
    cuda.memcpy_dtoh(size_arr, next_frontier_size_gpu)
    return int(size_arr[0])

def multi_seed_bfs_iteration(seeds,
                            rowptr_gpu, colidx_gpu,
                            assignments_gpu,
                            num_nodes,
                            bfs_kernel,
                            block_size):
    """
    BFS expansions from multiple seeds in parallel *until no new nodes* are assigned.
    seeds => list of (seed_node, cluster_id).
    """
    num_nodes_4bytes = num_nodes * 4

    # allocate device arrays for BFS
    d_frontier_nodes  = cuda.mem_alloc(num_nodes_4bytes)
    d_frontier_clusts = cuda.mem_alloc(num_nodes_4bytes)
    d_frontier_size   = cuda.mem_alloc(4)

    d_next_frontier_nodes  = cuda.mem_alloc(num_nodes_4bytes)
    d_next_frontier_clusts = cuda.mem_alloc(num_nodes_4bytes)
    d_next_frontier_size   = cuda.mem_alloc(4)

    # set initial frontier
    seed_nodes    = np.array([s[0] for s in seeds], dtype=np.int32)
    seed_clusters = np.array([s[1] for s in seeds], dtype=np.int32)
    fs = np.int32(len(seeds))

    cuda.memcpy_htod(d_frontier_nodes,  seed_nodes)
    cuda.memcpy_htod(d_frontier_clusts, seed_clusters)
    cuda.memcpy_htod(d_frontier_size,   fs)

    while True:
        next_size = multi_seed_bfs_expansion(
            rowptr_gpu, colidx_gpu,
            assignments_gpu, num_nodes,
            d_frontier_nodes, d_frontier_clusts, d_frontier_size,
            d_next_frontier_nodes, d_next_frontier_clusts, d_next_frontier_size,
            bfs_kernel, block_size
        )
        if next_size == 0:
            break

        # swap frontiers
        cuda.memcpy_dtod(d_frontier_nodes,  d_next_frontier_nodes,  next_size*4)
        cuda.memcpy_dtod(d_frontier_clusts, d_next_frontier_clusts, next_size*4)
        cuda.memcpy_htod(d_frontier_size, np.int32(next_size))

    # free BFS device arrays
    d_frontier_nodes.free()
    d_frontier_clusts.free()
    d_frontier_size.free()
    d_next_frontier_nodes.free()
    d_next_frontier_clusts.free()
    d_next_frontier_size.free()

def _merge_singleton_clusters(assignments, adjacency_matrix):
    """
    Merge 1-node clusters if they have neighbors in other clusters,
    to avoid orphan clusters.
    """
    from collections import defaultdict
    cluster_map= defaultdict(list)
    for node_idx,cid in enumerate(assignments):
        cluster_map[cid].append(node_idx)

    for cid,node_list in list(cluster_map.items()):
        if len(node_list)==1:
            u= node_list[0]
            st= adjacency_matrix.indptr[u]
            en= adjacency_matrix.indptr[u+1]
            neighbors= adjacency_matrix.indices[st:en]
            if len(neighbors)>0:
                neighbor_cids= assignments[neighbors]
                diff= [nc for nc in neighbor_cids if nc!=cid]
                if diff:
                    new_cid= diff[0]
                    assignments[u]= new_cid
                    cluster_map[cid].remove(u)
                    cluster_map[new_cid].append(u)

def create_clusters_with_ghosts_gpu(adjacency_matrix, kernel_manager, feature_matrix=None,
                                    block_size=128, seeds_per_batch=4):
    """
    Multi-seed BFS + ghost + merges singletons => returns list of clusters.
    """
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix= adjacency_matrix.tocsr()

    num_nodes= adjacency_matrix.shape[0]
    if feature_matrix is None:
        feature_matrix= sp.eye(num_nodes, format='csr')

    BFS_module= SourceModule(MULTI_SEED_BFS_KERNEL, no_extern_c=True)
    bfs_kernel= BFS_module.get_function("multi_seed_bfs_kernel")

    # copy adjacency to device
    d_row_ptr= cuda.mem_alloc(adjacency_matrix.indptr.nbytes)
    d_col_idx= cuda.mem_alloc(adjacency_matrix.indices.nbytes)
    cuda.memcpy_htod(d_row_ptr, adjacency_matrix.indptr)
    cuda.memcpy_htod(d_col_idx, adjacency_matrix.indices)

    assignments= np.full(num_nodes, -1, dtype=np.int32)
    d_assignments= cuda.mem_alloc(assignments.nbytes)
    cuda.memcpy_htod(d_assignments, assignments)

    degrees= np.diff(adjacency_matrix.indptr)

    # BFS expansions
    cluster_id=0
    while True:
        unassigned= np.where(assignments==-1)[0]
        if len(unassigned)==0:
            break

        deg_unassigned= degrees[unassigned]
        sorted_idxs= np.argsort(-deg_unassigned)
        k= min(seeds_per_batch, len(unassigned))
        chosen= unassigned[sorted_idxs[:k]]
        if k==0:
            break

        # assign each chosen node => new cluster_id
        for node in chosen:
            assignments[node]= cluster_id
            cluster_id+=1

        cuda.memcpy_htod(d_assignments, assignments)

        # build BFS seeds => (node, cluster_id)
        seeds= [(int(chosen[i]), int(assignments[chosen[i]])) for i in range(len(chosen))]

        multi_seed_bfs_iteration(
            seeds,
            d_row_ptr, d_col_idx,
            d_assignments,
            num_nodes,
            bfs_kernel,
            block_size
        )
        cuda.memcpy_dtoh(assignments, d_assignments)

    # merge singletons
    _merge_singleton_clusters(assignments, adjacency_matrix)
    cuda.memcpy_htod(d_assignments, assignments)

    # build cluster sets
    from collections import defaultdict
    cluster_map= defaultdict(list)
    for node_idx,cid in enumerate(assignments):
        cluster_map[cid].append(node_idx)
    clusters_primary= [sorted(v) for k,v in sorted(cluster_map.items()) if len(v)>0]

    # ghost expansions
    adj_indptr= adjacency_matrix.indptr
    adj_idx   = adjacency_matrix.indices

    assignment_dict={}
    for cid,c_nodes in enumerate(clusters_primary):
        for nd in c_nodes:
            assignment_dict[nd]=cid

    cluster_with_ghosts=[]
    for cid,c_nodes in enumerate(clusters_primary):
        cset= set(c_nodes)
        for nd in c_nodes:
            st= adj_indptr[nd]
            en= adj_indptr[nd+1]
            for e in range(st,en):
                nbr= adj_idx[e]
                if assignment_dict.get(nbr,-999)!=cid:
                    cset.add(nbr)
        cluster_with_ghosts.append(sorted(cset))

    d_row_ptr.free()
    d_col_idx.free()
    d_assignments.free()

    return cluster_with_ghosts

def create_clusters_metis_bfs_gpu(adjacency_matrix, kernel_manager, feature_matrix=None):
    """
    Replaces old METIS BFS => multi-seed BFS + ghost approach
    """
    try:
        clusters= create_clusters_with_ghosts_gpu(
            adjacency_matrix, kernel_manager, feature_matrix,
            block_size=128, seeds_per_batch=4
        )
        if not clusters:
            raise RuntimeError("Partitioning returned no clusters")
        return clusters
    except Exception as e:
        print(f"GPU clustering failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# --------------------------------------------------------------------
# Tile-based dense kernel + CSR spMM
# --------------------------------------------------------------------
TILE_DENSE_KERNEL = r'''
#define TILE_SIZE 16
extern "C" __global__ void tile_dense_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N, int K, int M
){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row= blockIdx.y*TILE_SIZE + threadIdx.y;
    int col= blockIdx.x*TILE_SIZE + threadIdx.x;

    float val=0.0f;
    for(int t=0; t< ( (K+TILE_SIZE-1)/TILE_SIZE ); t++){
        int Acol= t*TILE_SIZE + threadIdx.x;
        int Arow= row;

        int Brow= t*TILE_SIZE + threadIdx.y;
        int Bcol= col;

        if(Arow<N && Acol<K){
            As[threadIdx.y][threadIdx.x]= A[Arow*K + Acol];
        } else {
            As[threadIdx.y][threadIdx.x]=0.0f;
        }
        if(Brow<K && Bcol<M){
            Bs[threadIdx.y][threadIdx.x]= B[Brow*M + Bcol];
        } else {
            Bs[threadIdx.y][threadIdx.x]=0.0f;
        }
        __syncthreads();

        for(int i=0;i<TILE_SIZE;i++){
            val+= As[threadIdx.y][i]*Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if(row<N && col<M){
        C[row*M + col]= val;
    }
}
'''.strip()

SPARSE_MATMUL_KERNEL = r'''
extern "C" __global__
void sparse_matmul_csr(
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

    if(row<N && col<M){
        float sum=0.0f;
        int start= A_indptr[row];
        int end  = A_indptr[row+1];
        for(int e=start;e<end;e++){
            int kk= A_indices[e];
            float a_val= A_data[e];
            int bs= B_indptr[kk];
            int be= B_indptr[kk+1];

            // binary search
            int left= bs, right= be-1;
            while(left<=right){
                int mid=(left+right)>>1;
                int bc= B_indices[mid];
                if(bc==col){
                    sum+= a_val*B_data[mid];
                    break;
                }
                if(bc<col) left=mid+1;
                else right=mid-1;
            }
        }
        C[row*M + col]= sum;
    }
}
'''.strip()

class CUDAKernelManager:
    def __init__(self):
        self.context = None
        self.mod_dense = None
        self.mod_sparse= None
        self.tile_kernel= None
        self.sparse_kernel= None

    def init_context(self):
        if not self.context:
            self.context = cuda.Device(0).make_context()
            # compile dense + sparse
            self.mod_dense = SourceModule(TILE_DENSE_KERNEL, no_extern_c=True)
            self.tile_kernel= self.mod_dense.get_function("tile_dense_mm_kernel")

            self.mod_sparse= SourceModule(SPARSE_MATMUL_KERNEL, no_extern_c=True)
            self.sparse_kernel= self.mod_sparse.get_function("sparse_matmul_csr")

    def cleanup(self):
        if self.context:
            self.context.pop()
            self.context=None

class GPUPipeline:
    def __init__(self, kernel_manager):
        self.kernel_manager= kernel_manager
        self.kernel_manager.init_context()
        self.tile_size= 16

    def run_dense_tile_mm(self, A_dense, B_dense):
        N,K= A_dense.shape
        K2,M= B_dense.shape
        assert K==K2
        C_out= np.zeros((N,M), dtype=np.float32)

        dA= cuda.mem_alloc(A_dense.nbytes)
        dB= cuda.mem_alloc(B_dense.nbytes)
        dC= cuda.mem_alloc(C_out.nbytes)

        cuda.memcpy_htod(dA, A_dense)
        cuda.memcpy_htod(dB, B_dense)

        block=(self.tile_size,self.tile_size,1)
        grid=( (M+self.tile_size-1)//self.tile_size, (N+self.tile_size-1)//self.tile_size,1)

        self.kernel_manager.tile_kernel(
            dA, dB, dC,
            np.int32(N), np.int32(K), np.int32(M),
            block=block,grid=grid
        )
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(C_out, dC)
        dA.free(); dB.free(); dC.free()
        return C_out

    def run_sparse_matmul(self, subA, subB):
        N,K= subA.shape
        K2,F= subB.shape
        assert K==K2
        outC= np.zeros((N,F),dtype=np.float32)

        A_data= subA.data.astype(np.float32)
        A_idx = subA.indices.astype(np.int32)
        A_ptr = subA.indptr.astype(np.int32)

        B_data= subB.data.astype(np.float32)
        B_idx = subB.indices.astype(np.int32)
        B_ptr = subB.indptr.astype(np.int32)

        dA_data= cuda.mem_alloc(A_data.nbytes)
        dA_idx = cuda.mem_alloc(A_idx.nbytes)
        dA_ptr = cuda.mem_alloc(A_ptr.nbytes)
        dB_data= cuda.mem_alloc(B_data.nbytes)
        dB_idx = cuda.mem_alloc(B_idx.nbytes)
        dB_ptr = cuda.mem_alloc(B_ptr.nbytes)
        dC     = cuda.mem_alloc(outC.nbytes)

        cuda.memcpy_htod(dA_data,A_data)
        cuda.memcpy_htod(dA_idx, A_idx)
        cuda.memcpy_htod(dA_ptr, A_ptr)
        cuda.memcpy_htod(dB_data,B_data)
        cuda.memcpy_htod(dB_idx, B_idx)
        cuda.memcpy_htod(dB_ptr, B_ptr)

        block=(32,32,1)
        grid_x= (F+block[0]-1)//block[0]
        grid_y= (N+block[1]-1)//block[1]
        self.kernel_manager.sparse_kernel(
            dA_data, dA_idx, dA_ptr,
            dB_data, dB_idx, dB_ptr,
            dC,
            np.int32(N),np.int32(K), np.int32(F),
            block=block, grid=(grid_x,grid_y,1)
        )
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(outC, dC)
        dA_data.free(); dA_idx.free(); dA_ptr.free()
        dB_data.free(); dB_idx.free(); dB_ptr.free()
        dC.free()

        return outC

    def process_submatrix(self, subA, subF):
        N,K= subA.shape
        nnz= subA.nnz
        total_els= N*K
        density= nnz/(total_els+1e-9)
        # heuristic
        if (total_els<100_000) and (density>0.2):
            A_d= subA.toarray().astype(np.float32)
            B_d= subF.toarray().astype(np.float32)
            return self.run_dense_tile_mm(A_d, B_d)
        else:
            return self.run_sparse_matmul(subA, subF)

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    results_list= []
    kernel_manager= None
    pipeline= None
    try:
        kernel_manager= CUDAKernelManager()
        kernel_manager.init_context()
        pipeline= GPUPipeline(kernel_manager)

        for ginfo in graphs:
            idx= ginfo["index"]
            name= ginfo["name"]
            g  = ginfo["graph"]
            fm = ginfo["feature_matrix"]
            n  = ginfo["num_nodes"]
            s  = ginfo["sparsity"]
            graph_type= ginfo["type"]

            print(f"\n=== Graph {idx}: {name}, n={n}, s={s} ===")

            # Build adjacency
            adjacency= sp.lil_matrix((n,n),dtype=np.float32)
            for node in g.nodes:
                for nbr in g.neighbors(node):
                    adjacency[node,nbr]=1.0
            adjacency= adjacency.tocsr()
            feat= sp.csr_matrix(fm,dtype=np.float32)

            # 1) BFS partition
            bfs_start= time.time()
            clusters= create_clusters_metis_bfs_gpu(adjacency, kernel_manager, feat)
            bfs_end= time.time()
            bfs_time= bfs_end- bfs_start
            if not clusters:
                print("No clusters, skipping.")
                continue
            print(f"BFS partition => {len(clusters)} clusters, took {bfs_time:.4f}s")

            # 2) submatrix extraction
            def extract_submat(c):
                cset=set(c)
                neighbors=set()
                for nd in c:
                    st= adjacency.indptr[nd]
                    en= adjacency.indptr[nd+1]
                    neighbors.update(adjacency.indices[st:en])
                all_nodes= sorted(cset|neighbors)
                subA= adjacency[list(c),:][:,all_nodes].tocsr()
                subF= feat[all_nodes,:]
                return subA, subF, c

            ex_start= time.time()
            submatrices=[]
            for c in clusters:
                if len(c)<1:
                    continue
                sA, sF, cidx= extract_submat(c)
                if sA.nnz>0:
                    submatrices.append((sA,sF,cidx))
            ex_end= time.time()
            extract_time= ex_end- ex_start
            print(f"Extraction => {len(submatrices)} submatrices, {extract_time:.4f}s")

            # 3) spMM
            spmm_start= time.time()
            finalC= np.zeros((n, feat.shape[1]), dtype=np.float32)
            count = np.zeros(n, dtype=np.int32)
            spmm_results=[]
            for (subA, subF, node_list) in submatrices:
                outM= pipeline.process_submatrix(subA, subF)
                spmm_results.append((node_list, outM))
            spmm_end= time.time()
            spmm_time= spmm_end- spmm_start
            print(f"spMM => {spmm_time:.4f}s")

            # 4) merging is actually included in the spMM loop above
            # if you want to separate merging time, do so explicitly
            # but let's do an explicit "merge" measurement:
            merge_start= time.time()
            # we already partially merged, but let's do the "average"
            for (node_list, outM) in spmm_results:
                c = list(node_list)
                finalC[c] += outM
                count[c] += 1
            mask= (count!=0)
            finalC[mask]/= count[mask,None]
            merge_end= time.time()
            merge_time= merge_end- merge_start
            print(f"Merging => {merge_time:.4f}s")

            # 5) verification
            verify_start= time.time()
            ok= verify_result(finalC, adjacency, feat)
            verify_end= time.time()
            verify_time= verify_end- verify_start
            print(f"Verify => {ok}, {verify_time:.4f}s")

            if ok:
                results_list.append({
                    "graph_index": idx,
                    "graph_name": name,
                    "graph_type": graph_type,
                    "graph_type": "synthetic",
                    "num_nodes": n,
                    "sparsity": s,
                    "num_clusters": len(clusters),
                    "decomp": bfs_time,
                    "extract": extract_time,
                    "mult": spmm_time,
                    "merge": merge_time,
                    "verify_time": verify_time,
                    "is_correct": True
                })

    finally:
        if pipeline:
            pass  # pipeline doesn't hold its own context, kernel_manager does
        if kernel_manager:
            kernel_manager.cleanup()

    # Save JSON
    if results_list:
        try:
            if os.path.exists("gnn_results.json"):
                with open("gnn_results.json","r") as f:
                    try:
                        all_json= json.load(f)
                    except json.JSONDecodeError:
                        all_json= []
            else:
                all_json= []
            # update
            for rr in results_list:
                # remove any old entry
                all_json= [xx for xx in all_json if not(xx["graph_index"]==rr["graph_index"] and xx["method"]=="multi_seed_bfs_hybrid")]
                # set method
                rr["method"]= "multi_seed_bfs_hybrid"
                all_json.append(rr)

            with open("gnn_results.json","w") as f:
                json.dump(all_json, f, indent=4)
            print("Results appended => gnn_results.json")
        except Exception as e:
            print(f"Error saving results => {e}")

if __name__=="__main__":
    main()
