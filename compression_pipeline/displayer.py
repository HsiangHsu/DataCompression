'''
displayer.py

Short script to display the input and output data for debugging and testing.
'''

import numpy as np


data_in = np.load('data_in.npy')
data_out = np.load('data_out.npy')

assert data_in.shape == data_out.shape, \
    f'in: {data_in.shape}, out: {data_out.shape}'

for i in range(data_in.shape[0]):
    try:
        assert (data_in[i]==data_out[i]).all(), f'{i}'
    except:
        assert np.allclose(data_in, data_out, atol=1), \
        f'{data_in}\n\n{data_out}'

print(data_in, '\n\n', data_out, sep='')
