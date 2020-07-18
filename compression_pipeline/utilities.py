'''
utilities.py

This module includes various helper functions used by various modules
'''

from math import ceil, log2
import numpy as np


def valid_pixels_from_context_strategy(img_shape, relative_indices):
    '''
    Returns the minimum and maximum values for rows/columns to iterate over within an 
    image of dimension |img_shape| that can be sequentially decoded 
    in a linear scan pattern given sufficient initial context. 

    TODO(for more general raster scans): Because we do not currently support scan patterns 
    or initial context that is "ahead of"  (either to the right or below) a current pixel, 
    we require relative context indices to be negative-valued in the row or zero in the 
    current row but negative in the column.
    '''
    err_msg = "Impossible to satisfy passing initial context with these relative indices %r"
    assert np.all([index[0] < 0 or \
                  (index[0] == 0 and index[1] < 0) for index in relative_indices]), \
           err_msg % relative_indices
    min_x = abs(min([index[0] for index in relative_indices]))
    max_x = img_shape[1] - max(0, max([index[0] for index in relative_indices])) - 1
    min_y = abs(min([index[1] for index in relative_indices]))
    max_y = img_shape[0] - max(0, max([index[1] for index in relative_indices])) - 1
    return min_x, max_x, min_y, max_y


def get_valid_pixels_for_predictions(img_shape, current_context_indices, 
                                     prev_context_indices, return_tuples=False):
    '''
    Returns locations within an image of dimension |img_shape| that can be sequentially 
    decoded in a linear scan pattern given sufficient initial context and previous 
    context indices.

    Args:
        img_shape: tuple (x, y)
            dimensions of the ndarray for images in the dataset
        current_context_strategy: list of tuples (i, j)
            describes which pixels relative to a given location should be
            considered context in the current image
        prev_context_indices: list of tuples (i, j)
            describes which pixels relative to a given location should be
            considered context from prior images
        return_tuples: boolean
            whether the return value should be

    Returns:
        If |return_tuples|
            valid_pixels: list of (i, j) tuples
                indices from the range [min_x, max_x] x [min_y, max_y]
        Else
            (min_x, max_x, min_y, max_y): 4-tuple of ints
                inclusive indices for rows/columns that are valid to predict 
                given sufficient context
    '''
    min_x_cur, max_x_cur, min_y_cur, max_y_cur = valid_pixels_from_context_strategy(
                                                    img_shape, current_context_indices)
    min_x_prev, max_x_prev, min_y_prev, max_y_prev = valid_pixels_from_context_strategy(
                                                        img_shape, prev_context_indices)
    min_x = max(min_x_cur, min_x_prev)
    max_x = min(max_x_cur, max_x_prev)
    min_y = max(min_y_cur, min_y_prev)
    max_y = min(max_y_cur, max_y_prev)
    if return_tuples:
        valid_pixels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                valid_pixels += [(x, y)]
        return valid_pixels
    return min_x, max_x, min_y, max_y


def name_to_context_pixels(name):
    if name == 'DAB':
        return [(0, -1), (-1, -1), (-1, 0)]
    if name == 'DABC':
        return [(0, -1), (-1, -1), (-1, 0), (-1, 1)]
    return None


def convert_predictions_to_pixels(predictions, dtype):
    '''
    Casts the output of a model's |predictions| into pixel values that fit
    in |dtype|.
    '''
    minval, maxval = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(predictions, minval, maxval).astype(dtype)


def find_dtype(n):
    '''
    Finds the smallest numpy datatype given a maximum value that must be
    representable

    Args:
        n: int
            maximum value to represent

    Returns:
        dtype: numpy dtype
            smallest numpy dtype for n
    '''

    sizes = [8, 16, 32, 64]
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    valid_sizes = [sz for sz in sizes if sz >= int(ceil(log2(n)))]
    dtypes_index = sizes.index(min(valid_sizes))

    return dtypes[dtypes_index]


def readint(f, n):
    '''
    Reads in bytes from a file as an int

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
