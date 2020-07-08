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
    
    x_src = data[np.random.choice(data.shape[0])]
    x_target = data[np.argmax(np.linalg.norm(x_src - data, axis=1))]
    print("x_target is %05f from x_src" % np.linalg.norm(x_target - x_src))
    while len(data) > 0:
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1).fit(data)
        neighbor_inds = knn.kneighbors(np.array([x_src]), return_distance=False).flatten().astype('int')
        k_nearest = np.array([data[i] for i in neighbor_inds])  # includes |x_src|
        x_end = k_nearest[np.argmin(np.linalg.norm(x_target - k_nearest, ord=2, axis=1))]

        K = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1).fit(k_nearest)
        complete_subgraph = K.kneighbors_graph(n_neighbors=k-1, mode='distance').toarray()
        complete_subgraph = np.triu(complete_subgraph)
        TSP = christofides.compute(complete_subgraph)
        order += TSP['Christofides_Solution']
        



    
