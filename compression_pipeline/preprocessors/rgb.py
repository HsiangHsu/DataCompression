'''
preprocessors/rgb.py

This module contains the RGB layering preprocessor.
'''


def rgb_pre(data, rows, cols):
    '''
    RGB reshaping preprocessor

    Flat image data is reshaped into 3 layers, with each layer corresponding
    to an RGB color channel

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


def rgb_post(decomp):
    '''
    RGB reshaping postprocessor

    See docstring on the corresponding preprocessor for more information.

    Args:
        decomp: numpy array
            decompressed data, of shape
            (3, n_elements, rows, cols)

    Returns:
        post_data: numpy array
            postprocessed data, of shape
            (n_elements, 3*rows*cols)
    '''

    assert decomp.shape[0] == 3, f'invalid shape for RGB data: {decomp.shape}'

    post_data = decomp.swapaxes(0, 1)
    n_elements = post_data.shape[0]
    post_data = post_data.reshape(n_elements, np.prod(post_data.shape[1:]))

    return post_data
