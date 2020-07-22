'''
displayer.py

Short script for verifying that the input and output data are the same dataset,
even if they aren't in the same order.
'''

import numpy as np

data_in = np.load('data_in.npy')
data_out = np.load('data_out.npy')

if np.array_equiv(np.sort(data_in, axis=0), np.sort(data_out, axis=0)):
    print('\n\tCOMPRESSION CORRECT.\n')
else:
    print('\n\tCOMPRESSION INCORRECT.\n')
