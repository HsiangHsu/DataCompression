'''
encoders/delta_coo.py

This module contains the Delta-COOrdinate Matrix encoder
'''

from math import ceil, log2
import numpy as np
import pickle
from scipy.sparse import coo_matrix

from utilities import find_dtype, readint


def delta_coo_enc(compression, metadata, original_shape, args):
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

    with open('comp.out', 'wb') as f:
        f.write(metastream)
        f.write(datastream)
    with open('args.out', 'wb') as f:
        pickle.dump(args, f)


def delta_coo_dec(comp_file):
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
