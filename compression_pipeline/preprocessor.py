'''
preprocessor.py

This is the module responsible for processing and formatting the data before
compression.
'''


import numpy as np


def preprocess(data, args):
    '''
    Calls the appropriate preprocessor

    Args:
        data: numpy array
            data to be preprocessed
        preprocessor: string
            preprocessing algorithm to use
        kwargs: dict
            arguments to be passed to preprocessing algorithm

    Returns:
        preprocessed_data: numpy array
            preprocessed data
        element_axis: int
            index into data.shape for n_elements
    '''

    preprocessor = args.pre

    if preprocessor == 'sqpatch':
        return sqpatch(data, args.psz)
    if preprocessor == 'rgb':
        return rgb(data, args.rgbr, args.rgbc)
    if preprocessor == 'rgb-sqpatch':
        rgb_data, _ = rgb(data, args.rgbr, args.rgbc)
        return sqpatch(rgb_data, args.psz)


def sqpatch(data, patch_sz):
    '''
    Square patching preprocessor

    Square images are cropped into axis-aligned sub-squares.

    Args:
        data: numpy array
            data to be preprocessed, of shape
            (n_elements, [n_layers], width, width)
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

    # If the n_layers dimension is not present in the original data
    # (e.g. if the image is a single grayscale layer), add in that dimension
    # for the sake of generality.
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])

    # Reformat data as (n_layers, n_elements, width, width)

    n_layers = data.shape[0]
    n_elements = data.shape[1]

    patched_data = np.empty((n_layers, n_elements, n_patches,
        patch_sz, patch_sz), dtype=data.dtype)

    for i in range(n_layers):
        for j in range(n_elements):
                patched_data[i][j] = square_crop(data[i][j],
                    patches_per_side, patch_sz)

    # Reformat patched_data as (n_layers, n_patches, n_elements, patch_sz**2)
    patched_data = patched_data.swapaxes(1,2)
    patched_data = patched_data.reshape(*patched_data.shape[:-2],
        patch_sz**2)

    # after preprocesssing, element_axis = 2
    return patched_data, 2


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


def rgb(data, rows, cols):
    '''
    RGB reshaping preprocessor

    Flat image data is reshaped into 3 layers, with each layer corresponding to
    an RGB color channel

    Args:
        data: numpy array
            data to be preprocessed, of shape
            (n_elements, 3*rows*cols)
        rows: int
            number of rows in each RGB image
        cols: int
            number of columns in each RGB image

    Returns:
        rgb_data: numpy array
            RGB data, of shape
            (3, n_elements, rows, cols)
        element_axis: int
            index into data.shape for n_elements
    '''

    assert len(data.shape) == 2, f'invalid shape for RGB data: {data.shape}'
    assert data.shape[1] == 3*rows*cols, 'invalid RGB width and/or cols'

    n_elements = data.shape[0]
    data = data.reshape(n_elements, 3, rows, cols)
    rgb_data = data.swapaxes(0, 1)

    return rgb_data, 1
