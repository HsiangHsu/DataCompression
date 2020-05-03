'''
compressor.py

This is the module responsible for compressing, encoding, and writing the
compressed data to disk.

Extra modules for implementing the compressors can be found in the
/compressors directory.
'''


from math import sqrt, ceil, log2
import numpy as np
import pickle
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix

from compressors.knn_mst import process_mst, pad_order

def compress(data, compressor, **kwargs):
    '''
    Calls the appropriate compressor

    Args:
        data: numpy array
            data to be compressed
        compressor: string
            compression algorithm to use
        kwargs: dict
            arguments to be passed to the compression algorithm

    Returns:
        compressed_data: numpy array
            compressed data
    '''

    if compressor == 'knn-mst':
        return knn_mst(data, kwargs['n_neighbors'],
            kwargs['metric'], kwargs['minkowski_p'])


def encode(compression, metadata, encoder, args, **kwargs):
    '''
    Calls the appropriate encoder

    The encoder will write the compression to disk.

    Args:
        compression: numpy array
            compressed data to be encoded
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
        encoder: string
            encoder to use
        args: dict
            all command line arguments passed to driver_compress.py
        kwargs: dict
            arguments to be passed to the encoder

    Returns:
        None
    '''

    if encoder == 'delta-coo':
        delta_coo(compression, metadata, args)


def knn_mst(data, n_neighbors, metric, minkowski_p):
    '''
    K Nearest Neighbors and Minimum Spanning Tree compressor

    The KNN graph is constructed and the MST is then calculated on that graph.
    The MST is then used to produce an ordering of the elements such that
    the distance between each element is minimized.

    Assumes that a patching preprocessor, such as sqpatch, has been used,
    although the entire image can be treated as a single patch if no patching
    is desired.

    Args:
        data: numpy array
            data to be compressed, of shape
            (n_layers, n_patches, n_elements, patch_element_size)
        n_neighbors: int
            number of neighbors used to build KNN graph
        metric: string
            distance metric used to build KNN graph
        minkowski_p: int
            parameter used for the Minkowski distance metric

    Returns:
        ordered_data: numpy array
            appropriately ordered data, of shape
            (n_layers, n_patches, n_elements, patch_element_size)
        inverse_orders: numpy array
            array of inverse permutations that returns the ordeded data to the
            original dataset order, of shape (n_layers, n_patches, n_elements)
    '''

    n_layers = data.shape[0]
    n_patches = data.shape[1]
    n_elements = data.shape[2]

    invor_dtype = find_dtype(n_elements)
    inverse_orders = np.empty((n_layers, n_patches, n_elements),
        dtype=invor_dtype)
    ordered_data = np.empty(data.shape, dtype=data.dtype)

    for i in range(n_layers):
        for j in range(n_patches):
            # Builds a separate KNN graph and MST for each patch on each layer
            knn_graph = kneighbors_graph(data[i][j], n_neighbors=n_neighbors,
                metric=metric, p=minkowski_p, mode='distance')
            mst = minimum_spanning_tree(knn_graph).toarray()

            order = process_mst(mst)
            order = pad_order(order, n_elements, data[i][j])

            ordered_data[i][j] = data[i][j][order]
            inverse_orders[i][j] = np.arange(len(order))[np.argsort(order)]

    return ordered_data, inverse_orders


def delta_coo(compression, metadata, args):
    '''
    Delta Vector and COOrdinate Matrix encoder

    The differences between adjacent elements are written to disk as a
    COO matrix to exploit the sparisty of the differences (deltas).

    Assumes that a patching preprocessor, such as sqpatch, has been used,
    although the entire image can be treated as a single patch if no patching
    is desired.

    Intended to be used after a smart ordering compressor, such as knn-mst, but
    this is not required.

    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_patches, n_elements, patch_element_size)
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader), of shape
            (n_layers, n_patches, n_elements); since this encoder is intended
            to be used with smart orderings, this will probably be inverse
            orderings of some sort
        args: dict
            all command line arguments passed to driver_compress.py

    Returns:
        None
    '''

    n_layers = compression.shape[0]
    n_patches = compression.shape[1]
    n_elements = compression.shape[2]
    element_dim = int(sqrt(compression.shape[3]))

    compression = compression.reshape(n_layers, n_patches, n_elements,
        element_dim, element_dim)

    # Number of bytes required to express the number of cells in a single
    # patch element
    element_size_bytes = ceil(int(ceil(log2(element_dim**2))) / 8)

    # Datatype to use for the COO row and col member arrays
    row_col_dtype = find_dtype(element_dim)

    # Bytestreams to be built and written
    metastream = b''
    datastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
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

            # Write initial element in dense format. Everything after is a
            # sequence of deltas working from this starting point.
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
        pickle.dump(args, f)


def find_dtype(n):
    '''
    Helper function for finding the smallest numpy datatype given a maximum
    value that must be representable

    Args:
        n: int
            maximum value to represent

    Returns:
        dtype: numpy dtype
            smallest numpy dtype for n
    '''

    sizes = [8, 16, 32, 64]
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    valid_sizes = [sz for sz in sizes if sz >= int(ceil(log2(n)))]
    dtypes_index = sizes.index(min(valid_sizes))

    return dtypes[dtypes_index]
