'''
dct.py

This module contains the Discrete Cosine Transform preprocessor.

NB: This module is may not be fully correct or complete. It should be tested
before being used.
'''

import numpy as np
from scipy.fft import dctn, idctn

def dct_pre(data):
    '''
    DCT Preprocessor

    2D images are transformed using the Discrete Cosine Transform

    Args:
        data: numpy array
            data to be preprocessed, of shape
            ([n_layers], n_elements, rows, columns)

    Returns:
        dct_data: numpy array
            preprocessed data, of shape
            ([n_layers], n_elements, rows, columns)
        element_axis: int
            index into data.shape for n_elements
    '''

    assert data.ndim == 3 or data.ndim == 4, \
        f'invalid shape for DCT: {data.shape}'

    if data.ndim == 3:
        element_axis = 0
    else:
        element_axis = 1

    transform = \
        dctn(data, axes=[-1,-2], norm='ortho', workers=-1).astype(np.int)
    inverse = \
        idctn(transform, axes=[-1,-2], norm='ortho', workers=-1).clip(0).astype(np.uint8)
    assert np.allclose(data, inverse, atol=4)

    return dctn(data, axes=[-1,-2], norm='ortho', workers=-1), element_axis

def dct_post(decomp):
    '''
    DCT Postprocessor

    See docstring on the corresponding preprocessor for more information

    Args:
        decomp: numpy array
            data to be preprocessed, of shape
            ([n_layers], n_elements, rows, columns)

    Returns:
        post_data: numpy array
            postprocessed data, of shape
            ([n_layers], n_elements, rows, columns)
    '''

    return idctn(decomp, axes=[-1,-2], norm='ortho', workers=-1)
