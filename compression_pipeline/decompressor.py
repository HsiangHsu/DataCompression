'''
decompressor.py

This is the module responsible for decompressing, decoding, and writing the
decompressed data to disk.
'''


from math import log2, ceil, sqrt
import numpy as np
from scipy.sparse import coo_matrix


def decode(comp_file, decoder):
    '''
    Calls the appropriate decoder

    Args:
        comp_file: string
            filepath to compressed data
        decoder: string
            decoder to use

    Returns:
        (decompression, metadata) : (numpy array, numpy array) tuple
            metadata is for compression (not necessarily the same metadata
            that is returned by the loader)
    '''

    if decoder == 'delta-coo':
        return delta_coo(comp_file)


def decompress(compression, metadata, decompressor):
    '''
    Calls the appropriate decompressor

    Args:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
        decompressor: string
            decompressor to use

    Returns:
        decompression: numpy array
            decompressed data
    '''

    if decompressor == 'knn-mst':
        return knn_mst(compression, metadata)

    if comp_args['pre'] == 'sqpatch':
        decompression = sqpatch(compression)


def delta_coo(comp_file):
    '''
    Delta Vector and COOrdinate Matrix decoder

    See docstring on the corresponding encoder for more information.

    Args:
        comp_file: string
            path to encoded compression

    Returns:
        (compression, metadata): (numpy array, numpy array) tuple
            compressed data and its corresponding metadata;
            compression is of the shape
            (n_layers, n_patches, n_elements, patch_element_size);
            the metadata is probably an inverse ordering of some sort
    '''

    f = open(comp_file, 'rb')

    # Read in sizing and and datatype metadata to reconstruct arrays.
    n_layers = readbytes(f, 4)
    n_patches = readbytes(f, 4)
    n_elements = readbytes(f, 4)
    element_dim = readbytes(f, 4)
    comp_dtype = np.dtype(chr(readbytes(f, 1)))
    dsz = comp_dtype.itemsize
    meta_dtype = np.dtype(chr(readbytes(f, 1)))
    rc_dtype = np.dtype(chr(readbytes(f, 1)))
    rcsz = rc_dtype.itemsize

    element_size_bytes = ceil(int(ceil(log2(element_dim**2))) / 8)

    compression = np.empty((n_layers, n_patches, n_elements,
        element_dim, element_dim), dtype=comp_dtype)
    metadata = np.empty((n_layers, n_patches, n_elements),
        dtype=meta_dtype)


    for i in range(n_layers):
        for j in range(n_patches):
            for k in range(n_elements):
                metadata[i][j][k] = readbytes(f, meta_dtype.itemsize)

    for i in range(n_layers):
        for j in range(n_patches):
            # Read initial dense element
            for k in range(element_dim):
                for l in range(element_dim):
                    compression[i][j][0][k][l] = readbytes(f, dsz)

            # Read subsequent deltas as COOs and then convert to dense
            for k in range(1, n_elements):
                coo_size = readbytes(f, element_size_bytes)
                coo_data = np.frombuffer(f.read(dsz*coo_size),
                    dtype=comp_dtype)
                coo_row = np.frombuffer(f.read(rcsz*coo_size), dtype=rc_dtype)
                coo_col = np.frombuffer(f.read(rcsz*coo_size), dtype=rc_dtype)
                coo = coo_matrix((coo_data, (coo_row, coo_col)),
                    shape=(element_dim, element_dim))
                compression[i][j][k] = coo.todense() + compression[i][j][k-1]

    f.close()

    return compression, metadata


def knn_mst(compression, inverse_orders):
    '''
    K Nearest Neighbors and Minimum Spanning Tree compressor

    See docstring on the corresponding compressor for more information.

    Args:
        compression: numpy array
            compressed data, of shape
            (n_layers, n_patches, n_elements, patch_element_size)
        inverse_orders: numpy array
            inverse permutations to return the ordered data to the original
            dataset order, of shape (n_layers, n_patches, n_elements)

    Returns:
        decompression: numpy array
            decompressed data, of shape
            (n_layers, n_patches, n_elements, patch_element_size)
    '''

    n_layers = compression.shape[0]
    n_patches = compression.shape[1]

    for i in range(n_layers):
        for j in range(n_patches):
            compression[i][j] = compression[i][j][inverse_orders[i][j]]

    return compression


def readbytes(f, n):
    '''
    Helper function for reading in bytes from a file

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
