'''
postprocessor.py

This is the module responsible for postprocessing and formatting the data
after decompression.
'''


from math import sqrt
import numpy as np


def postprocess(decompression, postprocessor):
    '''
    Calls the appropriate postprocessor.

    Args:
        decompression: numpy array
            decompressed data
        decompressor: string
            decompressor to use

    Returns:
        postprocessed_data: numpy array
            postprocessed data
    '''

    if postprocessor == 'sqpatch':
        return sqpatch(decompression)


def sqpatch(decompression):
    '''
    Square patching preprocessor

    See docstring on the corresponding preprocessor for more information.

    Args:
        decompression: numpy array
            decompressed data, of shape
            (n_layers, n_patches, n_elements, patch_element_size)

    Returns:
        postprocessed_data: numpy array
            postprocessed data, of shape
            (n_elements, [n_layers], element_dim, element_dim)
    '''
    n_layers = decompression.shape[0]
    n_patches = decompression.shape[1]
    n_elements = decompression.shape[2]
    element_dim = decompression.shape[3]
    patch_dim = int(sqrt(n_patches))
    decomp_element_dim = patch_dim*element_dim

    postprocessed_data = np.empty((n_elements, n_layers, decomp_element_dim,
        decomp_element_dim), dtype=decompression.dtype)

    for n in range(n_elements):
        for i in range(n_layers):
            for j in range(patch_dim):
                for k in range(patch_dim):
                    postprocessed_data[n][i][j*element_dim:(j+1)*element_dim,
                        k*element_dim:(k+1)*element_dim] = \
                        decompression[i][j*patch_dim+k][n]

    return postprocessed_data.squeeze()
