#!/usr/bin/env python3

'''
displayer.py

Script to display the input and output data for debugging and testing.
'''

import numpy as np
from matplotlib import pyplot as plt


data_in = np.load('data_in.npy')
data_out = np.load('data_out.npy')

assert data_in.shape == data_out.shape, \
    f'in: {data_in.shape}, out: {data_out.shape}'

plt.imshow(data_out[0])
plt.show()
