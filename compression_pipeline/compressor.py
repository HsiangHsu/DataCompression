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
from encoders.huffman import get_freqs, huffman_encode


def compress(data, element_axis, compressor, **kwargs):
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
            compressed data, of shape (n_layers, n_elements, n_points)
        metadata: numpy array
            metadata corresponding to each layer in the compression
        original_shape: tuple
            shape of original data
    '''

    if compressor == 'knn-mst':
        return knn_mst(data, element_axis, kwargs['n_neighbors'],
            kwargs['metric'], kwargs['minkowski_p'])


def encode(compression, metadata, original_shape, encoder, args,
    **kwargs):
    '''
    Calls the appropriate encoder

    The encoder will write the compression to disk.

    Args:
        compression: numpy array
            compressed data to be encoded
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
        original_shape: tuple
            shape of original data
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
        delta_coo(compression, metadata, original_shape, args)
    elif encoder == 'delta-huff':
        delta_huff(compression, metadata, original_shape, args)


def knn_mst(data, element_axis, n_neighbors, metric, minkowski_p):
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
        # Builds a separate KNN graph and MST for each patch on each layer
        knn_graph = kneighbors_graph(data[i], n_neighbors=n_neighbors,
            metric=metric, p=minkowski_p, mode='distance')
        mst = minimum_spanning_tree(knn_graph).toarray()
        order = process_mst(mst)
        order = pad_order(order, n_elements, data[i])

        ordered_data[i] = data[i][order]
        inverse_orders[i] = np.arange(len(order))[np.argsort(order)]

    return ordered_data, inverse_orders, original_shape


def delta_coo(compression, metadata, original_shape, args):
    '''
    Delta Vector and COOrdinate Matrix encoder

    The differences between adjacent elements are written to disk as a
    COO matrix to exploit the sparisty of the differences (deltas).

    Intended to be used after a smart ordering compressor, such as knn-mst,
    but this is not required.

    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_elements, n_points)
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader), of shape
            (n_layers, n_elements); since this encoder is intended
            to be used with smart orderings, this will probably be inverse
            orderings of some sort
        original_shape: tuple
            shape of original data
        args: dict
            all command line arguments passed to driver_compress.py

    Returns:
        None
    '''

    n_layers = compression.shape[0]
    n_elements = compression.shape[1]
    n_points = compression.shape[2]

    # Number of bytes required to express n_points
    n_points_bytes = ceil(int(ceil(log2(n_points))) / 8)

    # Datatype to use for the COO row and col member arrays
    row_col_dtype = find_dtype(n_points)

    # Bytestreams to be built and written
    metastream = b''
    datastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
    metastream += n_layers.to_bytes(4, 'little')
    metastream += n_elements.to_bytes(4, 'little')
    metastream += n_points.to_bytes(4, 'little')
    metastream += ord(compression.dtype.char).to_bytes(1, 'little')
    metastream += ord(metadata.dtype.char).to_bytes(1, 'little')
    metastream += ord(np.dtype(row_col_dtype).char).to_bytes(1, 'little')
    metastream += len(original_shape).to_bytes(1, 'little')
    for i in range(len(original_shape)):
        metastream += original_shape[i].to_bytes(4, 'little')

    for i in range(n_layers):
        deltas = np.array([compression[i][j] - compression[i][j-1]
            for j in range(1, n_elements)])
        coos = [coo_matrix(delta) for delta in deltas]

        # Write initial element in dense format. Everything after is a
        # sequence of deltas working from this starting point.
        datastream += compression[i][0].tobytes()

        for coo in coos:
            datastream += coo.size.to_bytes(n_points_bytes, 'little')
            datastream += coo.data.tobytes()
            # datastream += coo.row.astype(row_col_dtype).tobytes()
            datastream += coo.col.astype(row_col_dtype).tobytes()

        metastream += metadata[i].tobytes()

    with open('comp_out', 'wb') as f:
        f.write(metastream)
        f.write(datastream)
    with open('args_out', 'wb') as f:
        pickle.dump(args, f)


def delta_huff(compression, metadata, original_shape, args):
    '''
    Delta Vector and Huffman encoder

    The differences between adjacent elements are written to disk using
    a Huffman code.

    Intended to be used after a smart ordering compressor, such as knn-mst,
    but this is not required.

    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_elements, n_points)
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader), of shape
            (n_layers, n_elements); since this encoder is intended
            to be used with smart orderings, this will probably be inverse
            orderings of some sort
        original_shape: tuple
            shape of original data
        args: dict
            all command line arguments passed to driver_compress.py

    Returns:
        None
    '''

    n_layers = compression.shape[0]
    n_elements = compression.shape[1]
    n_points = compression.shape[2]

    # Bytestreams to be built and written
    metastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
    metastream += n_layers.to_bytes(4, 'little')
    metastream += n_elements.to_bytes(4, 'little')
    metastream += n_points.to_bytes(4, 'little')
    metastream += ord(compression.dtype.char).to_bytes(1, 'little')
    metastream += ord(metadata.dtype.char).to_bytes(1, 'little')
    metastream += len(original_shape).to_bytes(1, 'little')
    for i in range(len(original_shape)):
        metastream += original_shape[i].to_bytes(4, 'little')
    metastream += metadata.tobytes()

    f = open('comp_out', 'wb')
    f.write(metastream)

    for i in range(n_layers):
        bitstream = ''
        deltas = np.array([compression[i][j] - compression[i][j-1]
            for j in range(1, n_elements)])

        # Generate Huffman code for deltas on this layer
        freqs = get_freqs(deltas.flatten())
        code = huffman_encode(freqs)
        codestream = pickle.dumps(code)
        f.write(len(codestream).to_bytes(4, 'little'))
        f.write(codestream)

        # Generate Huffman coded bitstream for data on this layer
        for delta in deltas:
            for point in delta:
                bitstream += code[point]
        padded_length = 8*ceil(len(bitstream)/8)
        padding = padded_length - len(bitstream);
        bitstream = f'{bitstream:0<{padded_length}}'
        bytestream = [int(bitstream[8*j:8*(j+1)], 2)
            for j in range(len(bitstream)//8)]

        # The first element is not Huffman coded because it is not a delta
        f.write(len(bytestream).to_bytes(4, 'little'))
        f.write(padding.to_bytes(1, 'little'))
        f.write(bytes(bytestream))
        f.write(compression[i][0].tobytes())

    f.close()

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
