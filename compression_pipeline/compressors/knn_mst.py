'''
knn_mst.py

This module contains helper functions for implementing the KNN-MST compressor.
'''

import numpy as np
from scipy.spatial import distance

from datetime import timedelta
from timeit import default_timer as timer

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_tree, dijkstra, \
    depth_first_order, connected_components


def csr_to_edges(csr):
    edges = []
    for i in range(len(csr)):
        row = csr[i]
        row_edges = [[i, x] for x in range(len(row)) if row[x] != 0]
        edges.extend(row_edges)
    return edges

def create_tree(edges):
    tree = {}
    for v1, v2 in edges:
        tree.setdefault(v1, set()).add(v2)
        tree.setdefault(v2, set()).add(v1)
    return tree

def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def dfs_farthest(graph, start, return_longest_path_len=False):
    farthest = start
    max_dist = 0
    visited, stack = set(), [[start, 0]]
    while stack:
        vertex, dist = stack.pop()
        if vertex not in visited:
            if dist > max_dist:
                max_dist = dist
                farthest = vertex
            visited.add(vertex)
            stack.extend([vert, dist+1] for vert in graph[vertex] - visited)
    if return_longest_path_len:
        return farthest, max_dist
    return farthest

def dfs_path(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                return path + [next]
            else:
                stack.append((next, path + [next]))

def remove_path(edges, path):
    remaining_edges = []
    orphans = set()
    for v1, v2 in edges:
        if v1 in path and v2 in path:
            continue
        elif v1 in path:
            orphans.add(v2)
            continue
        elif v2 in path:
            orphans.add(v1)
            continue
        remaining_edges.append([v1, v2])
    for v1, v2 in remaining_edges:
        orphans = orphans -  {v1, v2}
    return remaining_edges, orphans

def separate_subtrees(forest, remaining_nodes):
    subtrees = []
    while remaining_nodes:
        subtree_nodes = dfs(forest, remaining_nodes.pop())
        subtree = {k:v for k,v in forest.items() if k in subtree_nodes}
        subtrees.append(subtree)
        remaining_nodes = remaining_nodes - subtree_nodes
    return subtrees

def tree_edges_to_order(tree, edges, return_longest_path_len=False):
    longest_path_len = -1
    order = []
    u = None
    try:
        u = dfs_farthest(tree, list(tree.keys())[0])
    except:
        return order
    v = None
    if return_longest_path_len:
        v, longest_path_len = dfs_farthest(tree, u, True)
    else:
        v = dfs_farthest(tree, u)
    path = dfs_path(tree, u, v)
    order.extend(path)
    remaining_edges, orphans = remove_path(edges, path)
    order.extend(orphans)
    subforest = create_tree(remaining_edges)
    subtrees = separate_subtrees(subforest, set(subforest.keys()))
    for subtree in subtrees:
        subtree_nodes = set(subtree.keys())
        relevant_edges = [[v1, v2] for v1, v2 in remaining_edges if
            v1 in subtree_nodes and v2 in subtree_nodes]
        order.extend(mst_to_order(subtree, relevant_edges))
    if return_longest_path_len:
        return order, longest_path_len
    return order

def process_mst(mst):
    edges = csr_to_edges(mst)
    tree = create_tree(edges)
    return tree_edges_to_order(tree, edges)

def pad_order(order, n, data):
    missing_idxs = np.setdiff1d(np.arange(n), order)
    if missing_idxs.shape[0] == n:
        return missing_idxs
    try:
        missing_data = np.unique(data[missing_idxs], axis=0)
    except:
        assert order.shape[0] == n
        return order

    for i in range(missing_data.shape[0]):
        insert_idx = np.where(np.all(data[order] == missing_data[i], axis=1))[0][0]
        match_idxs = np.intersect1d(np.where(np.all(data == missing_data[i], axis=1)),
            missing_idxs)
        order = np.insert(order, insert_idx, match_idxs)

    return order

    # for m in missing:
    #     indices = np.where(np.all(data == data[m], axis=1))[0]
    #     try:
    #         target = np.setdiff1d(indices, missing)[0]
    #     except:
    #         return indices
    #     insert_index = order.index(target)
    #     order.insert(insert_index, m)
    # return order

def generate_order(mst):
    nonzero = mst.nonzero()[0]
    order = np.empty(0, np.int32)
    if len(nonzero) == 0:
        return order
    edges_traversed = 0
    while True:
        start_idx = nonzero[0]
        df_tree = depth_first_tree(mst, start_idx, directed=False)
        edges_traversed += df_tree.nnz
        order = np.append(order, depth_first_order(df_tree, start_idx,
            return_predecessors=False))
        if edges_traversed == mst.nnz:
            return order
        else:
            nonzero = np.setdiff1d(nonzero, order)
