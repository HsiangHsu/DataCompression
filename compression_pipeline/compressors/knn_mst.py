'''
knn_mst.py

This module contains helper functions for implementing the KNN-MST compressor.
'''

import numpy as np
from scipy.sparse.csgraph import depth_first_tree, depth_first_order, \
    minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph

from datetime import timedelta
from timeit import default_timer as timer

from utilities import find_dtype


def knn_mst_comp(data, element_axis, n_neighbors, metric, minkowski_p):
    '''
    K Nearest Neighbors and Minimum Spanning Tree compressor

    The KNN graph is constructed and the MST is then calculated on that
    graph. The MST is then used to produce an ordering of the elements such
    that the distance between each element is minimized.

    Args:
        data: numpy array
            data to be compressed
        element_axis: int
            axis of data that whose length is the number of elements
        n_neighbors: int
            number of neighbors used to build KNN graph
        metric: string
            distance metric used to build KNN graph
        minkowski_p: int
            parameter used for the Minkowski distance metric

    Returns:
        ordered_data: numpy array
            appropriately ordered data, of shape
            (n_layers, n_elements, n_points)
        inverse_orders: numpy array
            array of inverse permutations that returns the ordered data to
            the original dataset order, of shape (n_layers, n_elements)
        original_shape: tuple
            shape of original data
    '''

    # reshape data into a 3D array: (n_layers, n_elements, n_points)
    original_shape = data.shape
    n_elements = data.shape[element_axis]
    data = data.reshape((-1, *data.shape[element_axis:]))
    data = data.reshape((*data.shape[:2], -1))
    n_layers = data.shape[0]

    invor_dtype = find_dtype(n_elements)
    inverse_orders = np.empty(data.shape[:2], dtype=invor_dtype)
    ordered_data = np.empty(data.shape, dtype=data.dtype)

    for i in range(n_layers):
        print(f'\tLayer {i}:')
        start = timer()

        # Builds a separate KNN graph and MST for each patch on each layer
        knn_graph = kneighbors_graph(data[i], n_neighbors=n_neighbors,
            metric=metric, p=minkowski_p, mode='distance', n_jobs=-1)

        end = timer()
        print(f'\tknn_graph in {timedelta(seconds=end-start)}.')
        start = timer()

        ### TODO overwrite=True
        mst = minimum_spanning_tree(knn_graph, overwrite=False)

        end = timer()
        print(f'\tmst in {timedelta(seconds=end-start)}.')
        start = timer()

        ordered_data[i], order = generate_order(data[i], n_elements, mst)
        inverse_orders[i] = np.arange(len(order))[np.argsort(order)]

        end = timer()
        print(f'\torder in {timedelta(seconds=end-start)}.\n')

    return ordered_data, inverse_orders, original_shape


def knn_mst_decomp(compression, inverse_orders, original_shape):
    '''
    K Nearest Neighbors and Minimum Spanning Tree compressor

    See docstring on the corresponding compressor for more information.

    Args:
        compression: numpy array
            compressed data, of shape
        inverse_orders: numpy array
            inverse permutations to return the ordered data to the original
            dataset order, of shape (n_layers, n_elements)
        original_shape: tuple
            shape of original data

    Returns:
        decompression: numpy array
            decompressed data, of shape
    '''

    n_layers = compression.shape[0]

    for i in range(n_layers):
        compression[i] = compression[i][inverse_orders[i]]

    compression = compression.reshape(original_shape)

    return compression


def generate_order(data, n_elements, mst):
    nonzero = np.unique(np.array(mst.nonzero()).ravel())
    if len(nonzero) == 0:
        return np.empty(0, np.uint32)

    min_total_distance = np.inf
    min_order = np.empty(0, np.uint32)

    for i in range(nonzero.size):
        # print()
        nonzero = np.unique(np.array(mst.nonzero()).ravel())
        nnz_shortened = False
        order = np.empty(0, np.uint32)
        edges_traversed = 0
        while True:
            # print("nnz: ", nonzero)
            if nnz_shortened:
                start_idx = nonzero[0]
            else:
                start_idx = nonzero[i]
            df_tree = depth_first_tree(mst, start_idx, directed=False)
            edges_traversed += df_tree.nnz
            order = np.append(order, depth_first_order(df_tree, start_idx,
                return_predecessors=False))
            if edges_traversed == mst.nnz:
                order = pad_order(order, n_elements, data)
                # print("order:\n", data[order])
                if len(order) < n_elements:
                    break
                total_distance = np.sum(np.abs(np.diff(
                    data[order].astype(np.int), axis=0)))
                # print("total_dist: ", total_distance)
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    min_order = order
                break
            else:
                nonzero = np.setdiff1d(nonzero, order)
                nnz_shortened = True

    assert min_order.size == n_elements, \
        "\nfailed to create full order -- try increasing nneigh\n"

    # print('\n\n\n')
    print(data[min_order])
    # print(min_total_distance)

    return data[min_order], min_order

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
        try:
            insert_idx = np.where(np.all(data[order] == missing_data[i],
                axis=1))[0][0]
        except:
            print("\npad_order failed -- try increasing nneigh\n")
            exit()
        match_idxs = np.intersect1d(np.where(np.all(data == missing_data[i],
            axis=1)), missing_idxs)
        order = np.insert(order, insert_idx, match_idxs)
    return order
