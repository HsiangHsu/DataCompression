'''
dct.py

This module contains the Discrete Cosine Transform preprocessor.
'''

import numpy as np
from scipy.fft import dctn

def dct_pre(data):
    '''
    DCT preprocessor

    Square images are cropped into axis-aligned sub-squares.

    Args:
        data: numpy array
            data to be preprocessed, of shape
            ([n_layers], n_elements, rows, columns)

    Returns:
        dct_data: numpy array
            preprocessed data, of shape
            (n_layers, n_elements, rows, columns)
        element_axis: int
            index into data.shape for n_elements
    '''

    return dctn(data, axes=[-1,-2], norm='ortho', workers=-1), 1
