'''
displayer.py

Short script for verifying that the input and output data are the same dataset,
even if they aren't in the same order.
'''

import numpy as np


data_in = np.load('udata_in.npy')
data_out = np.load('data_out.npy')

assert data_in.shape == data_out.shape, \
    f'in: {data_in.shape}, out: {data_out.shape}'

assert (data_in[np.lexsort(data_in.T)] == data_out[np.lexsort(data_out.T)]).all(), \
    'data_in and data_out differ'
