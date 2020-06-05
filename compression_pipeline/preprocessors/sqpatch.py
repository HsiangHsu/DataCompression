'''
preprocessors/sqpatch.py

This module contains the square patch preprocessor.
'''

import numpy as np


def sqpatch_pre(data, patch_sz):
    '''
    Square patching preprocessor

    Square images are cropped into axis-aligned sub-squares.

    Args:
        data: numpy array
            data to be preprocessed, of shape
            ([n_layers], n_elements, width, width)
        patch_sz: int
            desired patch width (the original image with must be an integer
            multiple of patch_sz)

    Returns:
        patched_data: numpy array
            preprocessed data, of shape
            (n_layers, n_patches, n_elements, patch_sz**2)
        element_axis: int
            index into data.shape for n_elements
    '''

    assert data.shape[-1] == data.shape[-2], 'elements must be square'
    assert data.shape[-1] % patch_sz == 0, ('element dimension must be ' +
        'an integer multiple of cropped dimension')

    patches_per_side = data.shape[-1] // patch_sz
    n_patches = patches_per_side**2

    # Reformat data as (n_layers, n_elements, width, width)
    if len(data.shape) == 3:
        data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
    n_layers = data.shape[0]
    n_elements = data.shape[1]

    patched_data = np.empty((n_layers, n_elements, n_patches,
        patch_sz, patch_sz), dtype=data.dtype)

    for i in range(n_layers):
        for j in range(n_elements):
                patched_data[i][j] = square_crop(data[i][j],
                    patches_per_side, patch_sz)

    # Reformat patched_data as (n_layers, n_patches, n_elements, patch_sz**2)
    patched_data = patched_data.swapaxes(1, 2)
    patched_data = patched_data.reshape(*patched_data.shape[:-2],
        patch_sz**2)

    # after preprocesssing, element_axis = 2
    return patched_data, 2


def sqpatch_post(decomp):
    '''
    Square patching postprocessor

    See docstring on the corresponding preprocessor for more information.

    Args:
        decomp: numpy array
            decompressed data, of shape
            (n_layers, n_patches, n_elements, n_points)

    Returns:
        post_data: numpy array
            postprocessed data, of shape
            ([n_layers], n_elements, sqrt(n_points)*sqrt(n_patches))
    '''

    n_layers = decomp.shape[0]
    n_patches = decomp.shape[1]
    n_elements = decomp.shape[2]
    n_points = decomp.shape[3]
    patch_width = int(sqrt(n_points))
    patches_per_side = int(sqrt(n_patches))
    width = patches_per_side * patch_width

    decomp = decomp.reshape(n_layers, n_patches, n_elements,
        patch_width, patch_width)
    post_data = np.empty((n_elements, n_layers, width, width),
        dtype=decomp.dtype)

    for n in range(n_elements):
        for i in range(n_layers):
            for j in range(patches_per_side):
                for k in range(patches_per_side):
                    post_data[n][i][j*patch_width:(j+1)*patch_width,
                        k*patch_width:(k+1)*patch_width] = \
                        decomp[i][j*patches_per_side+k][n]

    # Remove extra axes that may have been added by the postprocessor
    post_data = post_data.swapaxes(0, 1).squeeze()

    return post_data


def square_crop(element, patches_per_side, patch_sz):
    '''
    Helper function for sqpatch that does the actual cropping

    Args:
        element: numpy array
            single image to be cropped, of shape (width, width)
        patches_per_side: int
            number of patches to fit along a side of the original image
        patch_sz: int
            desired patch width

    Returns:
        cropped: numpy array
            array of patches, of shape
            (patches_per_side**2, patch_sz, patch_sz)
    '''

    assert patches_per_side*patch_sz == element.shape[0], \
        'Can\'t fit patches into image.'
    assert element.shape[0] == element.shape[1], \
        'Image must be square.'

    cropped = np.empty((patches_per_side**2, patch_sz, patch_sz))

    for i in range(patches_per_side):
        for j in range(patches_per_side):
            cropped[i*patches_per_side+j] = \
                element[i*patch_sz:(i+1)*patch_sz, j*patch_sz:(j+1)*patch_sz]

    return cropped
