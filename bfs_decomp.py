#!/usr/bin/env python3

"""
multi_seed_bfs_partition_ghosts.py

Creates a node partition of a graph via multi-seed BFS on the GPU, then
adds ghost nodes so edges on cluster boundaries appear in each relevant
cluster's subgraph.

STEPS:
1) Multi-seed BFS partition:
   - Each node is assigned exactly once (primary assignment).
   - Seeds chosen among highest-degree uncovered nodes for faster coverage.
   - An unassigned node is claimed by the first BFS frontier that touches it.
2) Ghost-node addition:
   - For each cluster, if a primary node has a neighbor assigned to a *different* cluster,
     we replicate ("ghost") that neighbor (and thus the boundary edge) in this cluster as well.
   - The node still belongs to exactly one primary cluster, but "ghost" copies appear in other
     clusters that need that boundary for GNN/spMM correctness.

As a result, each cluster is a union of:
   primary_nodes + ghost_nodes
And each boundary edge is in *both* clusters if its endpoints have different primary assignments.

Usage:
  python multi_seed_bfs_partition_ghosts.py
"""

import numpy as np
import scipy.sparse as sp

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# ---------------------------------------------------------------------
# 1) Multi-seed BFS kernel: each frontier entry is (node, cluster).
#    We attempt to assign neighbors if they are -1 (unassigned).
# ---------------------------------------------------------------------
MULTI_SEED_BFS_KERNEL = r'''
extern "C" {
__global__ void multi_seed_bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ assignments,     // cluster_assignments[node]
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
    
    if (node < 0 || node >= num_nodes) return;
    // Expand neighbors
    int start = row_ptr[node];
    int end   = row_ptr[node + 1];
    for (int e = start; e < end; e++) {
        int nbr = col_idx[e];
        if (nbr >= 0 && nbr < num_nodes) {
            // If unassigned => claim neighbor
            if (atomicCAS(&assignments[nbr], -1, cluster) == -1) {
                // Add neighbor to next frontier
                int idx = atomicAdd(next_frontier_size, 1);
                next_frontier_nodes[idx]    = nbr;
                next_frontier_clusters[idx] = cluster;
            }
        }
    }
}
} // extern "C"
'''.strip()


def multi_seed_bfs_expansion(
    rowptr_gpu, colidx_gpu,
    assignments_gpu,
    num_nodes: int,
    frontier_nodes_gpu,
    frontier_clusters_gpu,
    frontier_size_gpu,
    next_frontier_nodes_gpu,
    next_frontier_clusters_gpu,
    next_frontier_size_gpu,
    bfs_kernel,
    block_size: int
):
    """
    Perform one BFS expansion pass from the given multi-seed frontier.
    Returns the size of the next frontier after expansion.
    """
    size_arr = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(size_arr, frontier_size_gpu)
    frontier_size = int(size_arr[0])
    if frontier_size == 0:
        return 0

    zero_val = np.int32(0)
    cuda.memcpy_htod(next_frontier_size_gpu, zero_val)

    blocks = (frontier_size + block_size - 1)//block_size
    blocks = max(blocks, 1)

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

    cuda.memcpy_dtoh(size_arr, next_frontier_size_gpu)
    return int(size_arr[0])


def multi_seed_bfs_iteration(
    seeds,
    rowptr_gpu, colidx_gpu,
    assignments_gpu,
    num_nodes: int,
    bfs_kernel,
    block_size: int
):
    """
    Expand all (node, cluster) seeds in parallel BFS until no new nodes are assigned.
    'seeds' is a list of (seed_node, cluster_id).
    """
    # Frontier arrays
    d_frontier_nodes  = cuda.mem_alloc(num_nodes*4)
    d_frontier_clusts = cuda.mem_alloc(num_nodes*4)
    d_frontier_size   = cuda.mem_alloc(4)

    d_next_frontier_nodes  = cuda.mem_alloc(num_nodes*4)
    d_next_frontier_clusts = cuda.mem_alloc(num_nodes*4)
    d_next_frontier_size   = cuda.mem_alloc(4)

    # Load seeds into frontier
    frontier_nodes  = np.array([s[0] for s in seeds], dtype=np.int32)
    frontier_clusts = np.array([s[1] for s in seeds], dtype=np.int32)
    fsize = np.int32(len(seeds))

    cuda.memcpy_htod(d_frontier_nodes,  frontier_nodes)
    cuda.memcpy_htod(d_frontier_clusts, frontier_clusts)
    cuda.memcpy_htod(d_frontier_size,   fsize)

    while True:
        size_next = multi_seed_bfs_expansion(
            rowptr_gpu, colidx_gpu,
            assignments_gpu,
            num_nodes,
            d_frontier_nodes,
            d_frontier_clusts,
            d_frontier_size,
            d_next_frontier_nodes,
            d_next_frontier_clusts,
            d_next_frontier_size,
            bfs_kernel,
            block_size
        )
        if size_next == 0:
            break  # no new assignments

        # swap frontiers
        cuda.memcpy_dtod(d_frontier_nodes,  d_next_frontier_nodes,  size_next*4)
        cuda.memcpy_dtod(d_frontier_clusts, d_next_frontier_clusts, size_next*4)
        cuda.memcpy_htod(d_frontier_size, np.int32(size_next))

    # cleanup
    d_frontier_nodes.free()
    d_frontier_clusts.free()
    d_frontier_size.free()
    d_next_frontier_nodes.free()
    d_next_frontier_clusts.free()
    d_next_frontier_size.free()


# ---------------------------------------------------------------------
# Primary function to build multi-seed BFS partition + add ghost nodes
# ---------------------------------------------------------------------
def build_multi_seed_bfs_partition_with_ghosts(
    adjacency_matrix: sp.csr_matrix,
    block_size: int = 64,
    seeds_per_batch: int = 1
):
    """
    1) Partition nodes into clusters using multi-seed BFS.
       - Each uncovered iteration picks 'seeds_per_batch' new seeds among
         highest-degree uncovered nodes.
       - BFS expansions run in parallel for those seeds, assigning any unassigned nodes.
    2) Build ghost nodes:
       - For each cluster c, for each node u in c, if u has a neighbor v in a different
         cluster c' != c, replicate v in c as a ghost. That ensures edges (u,v) appear
         in c's local subgraph as well.

    Returns a dict of:
      "assignments": 1D array (num_nodes) => each node's primary cluster
      "clusters_primary": list of primary node lists (cluster->list of nodes)
      "clusters_with_ghosts": list of all node lists (primary + ghost)
    """
    if not sp.isspmatrix_csr(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.tocsr()

    num_nodes = adjacency_matrix.shape[0]
    if num_nodes == 0:
        return {
            "assignments": np.array([], dtype=np.int32),
            "clusters_primary": [],
            "clusters_with_ghosts": []
        }

    # Compile the kernel
    module = SourceModule(MULTI_SEED_BFS_KERNEL, no_extern_c=True)
    bfs_kernel = module.get_function("multi_seed_bfs_kernel")

    # Copy adjacency to GPU
    d_row_ptr = cuda.mem_alloc(adjacency_matrix.indptr.nbytes)
    d_col_idx = cuda.mem_alloc(adjacency_matrix.indices.nbytes)
    cuda.memcpy_htod(d_row_ptr, adjacency_matrix.indptr)
    cuda.memcpy_htod(d_col_idx, adjacency_matrix.indices)

    # assignments array
    assignments = np.full(num_nodes, -1, dtype=np.int32)
    d_assignments = cuda.mem_alloc(assignments.nbytes)
    cuda.memcpy_htod(d_assignments, assignments)

    # Precompute degrees for picking seeds
    degrees = np.diff(adjacency_matrix.indptr)

    # BFS expansions
    cluster_id = 0
    while True:
        uncovered = np.where(assignments == -1)[0]
        if len(uncovered) == 0:
            break

        # pick seeds_per_batch new seeds from highest-degree uncovered
        deg_uncovered = degrees[uncovered]
        sorted_idxs = np.argsort(-deg_uncovered)  # descending
        k = min(seeds_per_batch, len(uncovered))
        chosen = uncovered[sorted_idxs[:k]]

        # assign them new cluster IDs
        for node in chosen:
            assignments[node] = cluster_id
            cluster_id += 1
        cuda.memcpy_htod(d_assignments, assignments)

        # seeds = [(node, cluster_id) for each chosen node]
        seeds = [(int(chosen[i]), int(assignments[chosen[i]])) for i in range(len(chosen))]

        # expand BFS in parallel
        multi_seed_bfs_iteration(
            seeds,
            d_row_ptr,
            d_col_idx,
            d_assignments,
            num_nodes,
            bfs_kernel,
            block_size
        )
        # copy updated assignments
        cuda.memcpy_dtoh(assignments, d_assignments)

    # Build primary cluster sets
    # cluster IDs go from 0..(cluster_id-1)
    cluster_map = {}
    for node_idx, cid in enumerate(assignments):
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append(node_idx)

    clusters_primary = [cluster_map[k] for k in sorted(cluster_map.keys())]

    # ------------------------------------------------------------------
    # Build ghost nodes so boundary edges appear in both clusters
    # ------------------------------------------------------------------
    # For each cluster c, for each node u in c, check neighbors:
    # if neighbor v is assigned to different cluster c' != c,
    # we add v as a ghost in c.
    # Then cluster c's subgraph includes edges (u,v).
    # ------------------------------------------------------------------
    adjacency_indptr = adjacency_matrix.indptr
    adjacency_indices = adjacency_matrix.indices

    # We'll build a final "clusters_with_ghosts" array of sets
    # Then convert to list
    clusters_with_ghosts_sets = []
    for c_nodes in clusters_primary:
        clusters_with_ghosts_sets.append(set(c_nodes))  # start with primary nodes

    # We need a quick way to see the cluster of each node
    # already have "assignments[node] => cluster"
    # Then we iterate each cluster c, each primary node u
    #   neighbors = adjacency_indices[indptr[u]..indptr[u+1]]
    #   if neighbor's assignment != c => ghost
    for c_idx, c_nodes in enumerate(clusters_primary):
        c_set = clusters_with_ghosts_sets[c_idx]
        for u in c_nodes:
            start = adjacency_indptr[u]
            end   = adjacency_indptr[u+1]
            for e in range(start, end):
                nbr = adjacency_indices[e]
                c_nbr = assignments[nbr]
                if c_nbr != c_idx:
                    # add neighbor as ghost
                    c_set.add(nbr)

    clusters_with_ghosts = [sorted(list(s)) for s in clusters_with_ghosts_sets]

    # Cleanup
    d_row_ptr.free()
    d_col_idx.free()
    d_assignments.free()

    return {
        "assignments": assignments,
        "clusters_primary": clusters_primary,
        "clusters_with_ghosts": clusters_with_ghosts
    }


def main():
    import time

    # Create a random adjacency for demonstration
    n = 1000
    p = 0.001
    rng = np.random.default_rng(42)
    rows, cols = np.where(rng.random((n, n)) < p)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Make undirected if you wish
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()

    print(f"Graph has {n} nodes, {A.nnz} edges.")

    start = time.time()
    result = build_multi_seed_bfs_partition_with_ghosts(
        A, block_size=128, seeds_per_batch=4
    )
    end = time.time()

    assignments = result["assignments"]
    clusters_primary = result["clusters_primary"]
    clusters_ghosted = result["clusters_with_ghosts"]

    print(f"Partition + ghosting done in {end - start:.4f} s.")
    print(f"Primary cluster count = {len(clusters_primary)}")
    print(f"Ghosted cluster count = {len(clusters_ghosted)} (same length, but bigger sets).")

    # Check coverage
    covered_nodes = np.sum([len(c) for c in clusters_primary])
    print(f"Covered by primary alone: {covered_nodes} (some might overlap if seeds > 1).")
    covered_ghosted = np.sum([len(c) for c in clusters_ghosted])
    print(f"Covered by ghosted sets: {covered_ghosted} (ghost additions).")

    # Example: print cluster sizes
    # for i in range(min(5, len(clusters_ghosted))):
    #     print(f"Cluster {i} primary size={len(clusters_primary[i])}, ghosted size={len(clusters_ghosted[i])}")

if __name__ == "__main__":
    main()
