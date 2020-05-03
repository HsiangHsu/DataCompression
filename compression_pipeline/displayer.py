'''
displayer.py

Short script to display the input and output data for debugging and testing.
'''


import numpy as np


data_in = np.load('data_in.npy')
data_out = np.load('data_out.npy')

print(data_in, '\n\n', data_out)
