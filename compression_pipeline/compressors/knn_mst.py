import numpy as np
from scipy.spatial import distance

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

def dfs_farthest(graph, start):
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

def tree_edges_to_order(tree, edges):
    order = []
    try:
        u = dfs_farthest(tree, list(tree.keys())[0])
    except:
        return order
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
        order.extend(tree_edges_to_order(subtree, relevant_edges))
    return order

def process_mst(mst):
    edges = csr_to_edges(mst)
    tree = create_tree(edges)
    return tree_edges_to_order(tree, edges)

def pad_order(order, n, data):
    missing = np.setdiff1d(np.arange(n), order)
    for m in missing:
        indices = np.where(np.all(data == data[m], axis=1))[0]
        try:
            target = np.setdiff1d(indices, missing)[0]
        except:
            return indices
        insert_index = order.index(target)
        order.insert(insert_index, m)
    return order
