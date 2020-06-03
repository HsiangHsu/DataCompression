'''
knn_mst.py

This module contains helper functions for implementing the KNN-MST compressor.
'''

import numpy as np
from scipy.sparse.csgraph import depth_first_tree, depth_first_order

from datetime import timedelta
from timeit import default_timer as timer

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
            insert_idx = np.where(np.all(data[order] == missing_data[i], axis=1))[0][0]
        except:
            print("\npad_order failed -- try increasing nneigh\n")
        match_idxs = np.intersect1d(np.where(np.all(data == missing_data[i], axis=1)),
            missing_idxs)
        order = np.insert(order, insert_idx, match_idxs)

    return order
