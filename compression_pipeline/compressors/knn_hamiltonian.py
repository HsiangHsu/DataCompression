'''
knn_hamiltonian.py

This module contains helper functions for implementing the KNN-Hamiltonian compressor.
Author: Madeleine Barowsky
'''

import numpy as np
from scipy.sparse.csgraph import depth_first_tree, depth_first_order, \
    minimum_spanning_tree, connected_components
from sklearn.neighbors import NearestNeighbors, DistanceMetric

from datetime import timedelta
from timeit import default_timer as timer

from utilities import find_dtype
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching
import networkx as nx
import itertools


def knn_hamiltonian_comp(data, k):
    '''
    K Nearest Neighbors and Hamiltonian compressor (using Christofides' approximation)

    Args:
        data: numpy array
            data to be compressed
        k: int

    Returns:
        order: numpy array
            permutation of original |data|
        inverse_orders: numpy array
            array of inverse permutations that returns the ordered data to
            the original dataset order, of shape (1, n_elements)
        original_shape: tuple
            shape of original data
    '''
    assert k < len(data), "K neighbors must be less than number of points"
    original_shape = data.shape
    original_dtype = data.dtype
    n_elements = data.shape[0]

    # Cast to a signed type to avoid overflow when taking distances
    data = data.reshape(n_elements, -1).astype('int16')

    src_index = np.random.choice(data.shape[0])
    x_src = data[src_index]
    
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1).fit(data)
    knn_graph = knn.kneighbors_graph(n_neighbors=k-1, mode='distance')

    assert connected_components(knn_graph, directed=False, return_labels=False) == 1, "KNN is not connected; increase k"

    mst_mat = minimum_spanning_tree(knn_graph, overwrite=True)
    mst = nx.Graph()
    mat_data, indices, indptr = mst_mat.data, mst_mat.indices, mst_mat.indptr
    for i in range(mst_mat.shape[0]):
        for j in range(indptr[i], indptr[i+1]):
            mst.add_edge(i, indices[j], weight = mat_data[j])   
    
    odd_vertices = [i for i in mst.nodes if mst.degree[i] % 2 != 0]
    odd_degree_subgraph = mst.subgraph(odd_vertices).copy()
    
    for u in odd_degree_subgraph.nodes:
        for v in odd_degree_subgraph.nodes:
            if u != v:
                if not odd_degree_subgraph.has_edge(u, v):
                    odd_degree_subgraph.add_edge(u, v, weight=np.linalg.norm(data[u] - data[v], ord=2))
    
    matching = minimum_weight_full_matching(odd_degree_subgraph)
    added_edges = []
    multigraph = nx.MultiGraph(incoming_graph_data=mst)
    for pair in matching:
        if (pair, matching[pair]) not in added_edges and (matching[pair], pair) not in added_edges:
            multigraph.add_edge(pair, matching[pair],weight=odd_degree_subgraph[pair][matching[pair]])
            added_edges.append((pair, matching[pair]))
    
    euler_tour = nx.algorithms.euler.eulerian_circuit(multigraph, keys=True)
    order = []
    for u,v,k in euler_tour:
        if u not in order:
            order.append(u)

    return np.array([data[order].astype(original_dtype)]), np.array(order), original_shape
    



    
