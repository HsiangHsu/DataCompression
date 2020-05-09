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
            (n_layers, n_patches, n_elements, n_points)

    Returns:
        postprocessed_data: numpy array
            postprocessed data, of shape
            (n_elements, [n_layers], sqrt(n_points)*sqrt(n_patches))
    '''

    n_layers = decompression.shape[0]
    n_patches = decompression.shape[1]
    n_elements = decompression.shape[2]
    n_points = decompression.shape[3]
    patch_width = int(sqrt(n_points))
    patches_per_side = int(sqrt(n_patches))
    width = patches_per_side * patch_width

    decompression = decompression.reshape(n_layers, n_patches, n_elements,
        patch_width, patch_width)
    postprocessed_data = np.empty((n_elements, n_layers, width, width),
        dtype=decompression.dtype)

    for n in range(n_elements):
        for i in range(n_layers):
            for j in range(patches_per_side):
                for k in range(patches_per_side):
                    postprocessed_data[n][i][j*patch_width:(j+1)*patch_width,
                        k*patch_width:(k+1)*patch_width] = \
                        decompression[i][j*patches_per_side+k][n]

    # Remove extra axes that may have been added by the postprocessor
    postprocessed_data = postprocessed_data.squeeze()

    return postprocessed_data
