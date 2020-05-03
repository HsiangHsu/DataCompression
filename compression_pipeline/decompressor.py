from math import log2, ceil, sqrt
import numpy as np
from scipy.sparse import coo_matrix

def decompress(comp_file, comp_args):
    if comp_args['enc'] == 'delta-coo':
        compression, inverse_orders = delta_coo(comp_file)
    if comp_args['comp'] == 'knn-mst':
        decompression = knn_mst(compression, inverse_orders)
    if comp_args['pre'] == 'sqpatch':
        decompression = sqpatch(compression)
    return decompression

def delta_coo(comp_file):
    with open(comp_file, 'rb') as f:
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
        inverse_orders = np.empty((n_layers, n_patches, n_elements),
            dtype=meta_dtype)

        for i in range(n_layers):
            for j in range(n_patches):
                for k in range(n_elements):
                    inverse_orders[i][j][k] = readbytes(f, meta_dtype.itemsize)

        for i in range(n_layers):
            for j in range(n_patches):
                for k in range(element_dim):
                    for l in range(element_dim):
                        compression[i][j][0][k][l] = \
                            readbytes(f, dsz)
                for k in range(1, n_elements):
                    coo_size = readbytes(f, element_size_bytes)
                    coo_data = np.frombuffer(f.read(dsz*coo_size),
                        dtype=comp_dtype)
                    coo_row = np.frombuffer(f.read(rcsz*coo_size),
                        dtype=rc_dtype)
                    coo_col = np.frombuffer(f.read(rcsz*coo_size),
                        dtype=rc_dtype)
                    coo = coo_matrix((coo_data, (coo_row, coo_col)),
                        shape=(element_dim, element_dim))
                    compression[i][j][k] = coo.todense() + compression[i][j][k-1]

        compression = compression.reshape(n_layers, n_patches, n_elements,
            element_dim**2)

        return compression, inverse_orders

def knn_mst(compression, inverse_orders):
    n_layers = compression.shape[0]
    n_patches = compression.shape[1]

    for i in range(n_layers):
        for j in range(n_patches):
            compression[i][j] = compression[i][j][inverse_orders[i][j]]

    return compression

def sqpatch(compression):
    n_layers = compression.shape[0]
    n_patches = compression.shape[1]
    n_elements = compression.shape[2]
    element_dim = int(sqrt(compression.shape[3]))
    patch_dim = int(sqrt(n_patches))
    decomp_element_dim = patch_dim*element_dim

    decompression = np.empty((n_elements, n_layers, decomp_element_dim,
        decomp_element_dim), dtype=compression.dtype)

    for n in range(n_elements):
        for i in range(n_layers):
            for j in range(patch_dim):
                for k in range(patch_dim):
                    decompression[n][0][j*element_dim:(j+1)*element_dim,
                        k*element_dim:(k+1)*element_dim] = \
                        compression[i][j*patch_dim+k][n].reshape(2, 2)

    return decompression.squeeze()

def readbytes(f, n):
    return int.from_bytes(f.read(n), byteorder='little')
