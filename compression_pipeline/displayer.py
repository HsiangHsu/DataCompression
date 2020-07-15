'''
displayer.py

Short script to display the input and output data for debugging and testing.
'''

import numpy as np


data_in = np.load('data_in.npy')
data_out = np.load('data_out.npy')

assert data_in.shape == data_out.shape, \
    f'in: {data_in.shape}, out: {data_out.shape}'

print(data_in[np.lexsort(data_in.T)][-1], '\n\n', data_out[np.lexsort(data_out.T)][-1], sep='')
