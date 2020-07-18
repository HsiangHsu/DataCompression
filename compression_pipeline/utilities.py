'''
utilities.py

This module includes various helper functions used by various modules
'''

from math import ceil, log2
import numpy as np


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
