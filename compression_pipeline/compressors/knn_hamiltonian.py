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
from Christofides import christofides
import networkx as nx
import itertools


def knn_hamiltonian_comp(data, k):
    '''
    K Nearest Neighbors and Hamiltonian compressor

    Args:
        data: numpy array
            data to be compressed
        k: int

    Returns:
        order: numpy array
            permutation of original |data|
    '''
    assert k < len(data), "K neighbors must be less than number of points"
    order = []
    original_shape = data.shape
    n_elements = data.shape[0]
    data = data.reshape((-1, *data.shape[0:]))
    # Cast to a signed type to avoid overflow when taking distances
    data = data.reshape((*data.shape[:2], -1))[0].astype('int16')
    
    src_index = np.random.choice(data.shape[0])
    print("src_index=", src_index)
    x_src = data[src_index]
    
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1).fit(data)
    knn_graph = knn.kneighbors_graph(n_neighbors=k-1, mode='distance')

    assert connected_components(knn_graph, directed=False, return_labels=False) == 1, "KNN is not connected; increase k"

    mst = christofides._csr_gen_triples(minimum_spanning_tree(knn_graph, overwrite=True))
    odd_vertices = christofides._odd_vertices_of_MST(mst, n_elements)

    # Fill in edge weights we need
    bipartite_set = [set(i) for i in itertools.combinations(set(odd_vertices), len(odd_vertices)//2)]
    for vertex_set1 in bipartite_set:
        vertex_set1 = list(sorted(vertex_set1))
        vertex_set2 = []
        for vertex in odd_vertices:
            if vertex not in vertex_set1:
                vertex_set2.append(vertex)
        matrix = [[np.inf for j in range(len(vertex_set2))] for i in range(len(vertex_set1))]
        for i in range(len(vertex_set1)):
            for j in range(len(vertex_set2)):
                weight = np.linalg.norm(data[vertex_set1[i]] - data[vertex_set2[j]], ord=2)
                mst.add_edge(i, j, weight=weight)
    print("finished filling edge weights we need, time for christofides")
    # TODO make compute_from_mst more efficient to take in bipartite graphs since we've already partially computed
    TSP = christofides.compute_from_mst(None, mst, odd_vertices=odd_vertices)
    order += TSP['Christofides_Solution']
    print(order)
    



    
