'''
decompressor.py

This is the module responsible for decompressing, decoding, and writing the
decompressed data to disk.
'''


from math import log2, ceil, sqrt
import numpy as np
import pickle
from scipy.sparse import coo_matrix

from encoders.huffman import huffman_decode


def decode(comp_file, decoder):
    '''
    Calls the appropriate decoder

    Args:
        comp_file: string
            filepath to compressed data
        decoder: string
            decoder to use

    Returns:
        decompression: numpy array
            decompressed data
        metadata: numpy array
            metadata is for compression (not necessarily the same metadata
            that is returned by the loader)
    '''

    if decoder == 'delta-coo':
        return delta_coo(comp_file)
    elif decoder == 'delta-huff':
        return delta_huff(comp_file)


def decompress(compression, metadata, original_shape, decompressor):
    '''
    Calls the appropriate decompressor

    Args:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
        original_shape: tuple
            shape of original data
        decompressor: string
            decompressor to use

    Returns:
        decompression: numpy array
            decompressed data
    '''

    if decompressor == 'knn-mst':
        return knn_mst(compression, metadata, original_shape)


def delta_coo(comp_file):
    '''
    Delta Vector and COOrdinate Matrix decoder

    See docstring on the corresponding encoder for more information.

    Args:
        comp_file: string
            path to encoded compression

    Returns:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
    '''

    f = open(comp_file, 'rb')

    # Read in sizing and and datatype metadata to reconstruct arrays.
    n_layers = readint(f, 4)
    n_elements = readint(f, 4)
    n_points = readint(f, 4)
    comp_dtype = np.dtype(chr(readint(f, 1)))
    meta_dtype = np.dtype(chr(readint(f, 1)))
    rc_dtype = np.dtype(chr(readint(f, 1)))
    original_shape_len = readint(f, 1)
    shape_values = []
    for i in range(original_shape_len):
        shape_values.append(readint(f, 4))
    original_shape = tuple(shape_values)

    dsz = comp_dtype.itemsize
    rcsz = rc_dtype.itemsize

    element_size_bytes = ceil(int(ceil(log2(n_points))) / 8)

    compression = np.empty((n_layers, n_elements, n_points),
        dtype=comp_dtype)
    metadata = np.empty((n_layers, n_elements), dtype=meta_dtype)

    for i in range(n_layers):
        for j in range(n_elements):
            metadata[i][j] = readint(f, meta_dtype.itemsize)

    for i in range(n_layers):
        # Read initial dense element
        for j in range(n_points):
            compression[i][0][j] = readint(f, dsz)

        # Read subsequent deltas as COOs and then convert to dense
        for j in range(1, n_elements):
            coo_size = readint(f, element_size_bytes)
            coo_data = np.frombuffer(f.read(dsz*coo_size),
                dtype=comp_dtype)
            # coo_row = np.frombuffer(f.read(rcsz*coo_size), dtype=rc_dtype)
            coo_row = np.zeros(coo_size)
            coo_col = np.frombuffer(f.read(rcsz*coo_size), dtype=rc_dtype)
            coo = coo_matrix((coo_data, (coo_row, coo_col)),
                shape=(1, n_points))
            compression[i][j] = coo.todense() + compression[i][j-1]

    f.close()

    return compression, metadata, original_shape

def delta_huff(comp_file):
    '''
    Delta Vector and Huffman Encoding Decoder

    See docstring on the corresponding encoder for more information.

    Args:
        comp_file: string
            path to encoded compression

    Returns:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
    '''
    f = open(comp_file, 'rb')

    # Read in sizing and and datatype metadata to reconstruct arrays.
    n_layers = readint(f, 4)
    n_elements = readint(f, 4)
    n_points = readint(f, 4)
    comp_dtype = np.dtype(chr(readint(f, 1)))
    meta_dtype = np.dtype(chr(readint(f, 1)))
    original_shape_len = readint(f, 1)
    shape_values = []
    for i in range(original_shape_len):
        shape_values.append(readint(f, 4))
    original_shape = tuple(shape_values)

    data_size = comp_dtype.itemsize
    meta_size = meta_dtype.itemsize

    compression = np.empty((n_layers, n_elements, n_points), dtype=comp_dtype)
    metadata = np.empty((n_layers, n_elements), dtype=meta_dtype)

    for i in range(n_layers):
        for j in range(n_elements):
            metadata[i][j] = readint(f, meta_size)

    for i in range(n_layers):
        codestream_len = readint(f, 4)
        code = pickle.loads(f.read(codestream_len))
        bytestream_len = readint(f, 4)
        padding_bits = readint(f, 1)
        bitstream_len = bytestream_len*8 - padding_bits
        bytestream = f.read(bytestream_len)

        for j in range(n_points):
            compression[i][0][j] = readint(f, data_size)

        # Corner case where there is no difference across layer
        if bytestream_len == 0:
            for j in range(1, n_elements):
                compression[i][j] = compression[i][0]
            continue

        decodings = {v: k for k, v in code.items()}
        decoded_stream = huffman_decode(bytestream, bitstream_len, decodings)
        deltas = np.array(decoded_stream, dtype=comp_dtype)
        deltas = deltas.reshape(n_elements-1, n_points)

        for j in range(1, n_elements):
            compression[i][j] = compression[i][j-1] + deltas[j-1]

    return compression, metadata, original_shape


def knn_mst(compression, inverse_orders, original_shape):
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


def readint(f, n):
    '''
    Helper function for reading in bytes from a file as an int

    Args:
        f: open file
            file to read from
        n: int
            number of bytes to read

    Returns:
        d: int
            integer representation of the n bytes read
    '''

    return int.from_bytes(f.read(n), byteorder='little')
