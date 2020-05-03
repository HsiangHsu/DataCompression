from math import sqrt, ceil, log2
import numpy as np
import pickle
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix

from compressors.knn_mst import process_mst, pad_order

def compress(data, compressor, **kwargs):
    if compressor == 'knn-mst':
        return knn_mst(data, kwargs['n_neighbors'],
            kwargs['metric'], kwargs['minkowski_p'])

def encode(compression, metadata, encoder, args, **kwargs):
    if encoder == 'delta-coo':
        delta_coo(compression, metadata, args)

def knn_mst(data, n_neighbors, metric, minkowski_p):
    n_layers = data.shape[0]
    n_patches = data.shape[1]
    n_elements = data.shape[2]

    invor_dtype = find_dtype(n_elements)
    inverse_orders = np.empty((n_layers, n_patches, n_elements),
        dtype=invor_dtype)

    ordered_data = np.empty(data.shape, dtype=data.dtype)

    for i in range(n_layers):
        for j in range(n_patches):
            knn_graph = kneighbors_graph(data[i][j], n_neighbors=n_neighbors,
                metric=metric, p=minkowski_p, mode='distance')
            mst = minimum_spanning_tree(knn_graph).toarray()
            order = process_mst(mst)
            order = pad_order(order, n_elements, data[i][j])
            inverse_orders[i][j] = np.arange(len(order))[np.argsort(order)]
            ordered_data[i][j] = data[i][j][order]
    return ordered_data, inverse_orders

def delta_coo(compression, metadata, args):
    n_layers = compression.shape[0]
    n_patches = compression.shape[1]
    n_elements = compression.shape[2]
    element_dim = int(sqrt(compression.shape[3]))

    compression = compression.reshape(n_layers, n_patches, n_elements,
        element_dim, element_dim)

    element_size_bytes = ceil(int(ceil(log2(element_dim**2))) / 8)

    row_col_dtype = find_dtype(element_dim)

    metastream = b''
    datastream = b''

    metastream += n_layers.to_bytes(4, 'little')
    metastream += n_patches.to_bytes(4, 'little')
    metastream += n_elements.to_bytes(4, 'little')
    metastream += element_dim.to_bytes(4, 'little')
    metastream += ord(compression.dtype.char).to_bytes(1, 'little')
    metastream += ord(metadata.dtype.char).to_bytes(1, 'little')
    metastream += ord(np.dtype(row_col_dtype).char).to_bytes(1, 'little')

    for i in range(n_layers):
        for j in range(n_patches):
            deltas = np.array([compression[i][j][k] - compression[i][j][k-1]
                for k in range(1, n_elements)])
            coos = [coo_matrix(delta) for delta in deltas]
            datastream += compression[i][j][0].tobytes()
            for coo in coos:
                datastream += coo.size.to_bytes(element_size_bytes, 'little')
                datastream += coo.data.tobytes()
                datastream += coo.row.astype(row_col_dtype).tobytes()
                datastream += coo.col.astype(row_col_dtype).tobytes()
            metastream += metadata[i][j].tobytes()

    with open('comp_out', 'wb') as f:
        f.write(metastream)
        f.write(datastream)
    with open('args_out', 'wb') as f:
        pickle.dump(vars(args), f)

def find_dtype(n):
    sizes = [8, 16, 32, 64]
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    valid_sizes = [sz for sz in sizes if sz >= int(ceil(log2(n)))]
    dtypes_index = sizes.index(min(valid_sizes))
    return dtypes[dtypes_index]
