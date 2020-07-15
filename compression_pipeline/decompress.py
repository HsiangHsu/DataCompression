#!/usr/bin/env python3

'''
driver_decompress.py

This is the executable used to decompress a dataset.
The compressed dataset and the  arguments used for compression
are passed in as command-line arguments.
After parsing arguments, the work is delegated to the decompressor module.
'''

import argparse
import numpy as np
import pickle
import sys

from drivers import decompress, decode, postprocess

from datetime import timedelta
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('--data_in', type=str, default='comp.out',
    help='compressed data to decompress')
parser.add_argument('--args_in', type=str, default='args.out',
    help='arguments used to compress data')

args = parser.parse_args()

full_start = timer()

with open(args.args_in, 'rb') as f:
    comp_args = pickle.load(f)

start = timer()
compression, pre_metadata, comp_metadata, original_shape = decode(args.data_in,
    comp_args)
end = timer()
print(f'\ndecode in {timedelta(seconds=end-start)}.\n')

start = timer()
decomp_data = decompress(compression, comp_metadata, original_shape, comp_args)
end = timer()
print(f'decompress in {timedelta(seconds=end-start)}.\n')

start = timer()
if comp_args.pre:
    decomp_data = postprocess(decomp_data, pre_metadata, comp_args)
end = timer()
print(f'postprocess in {timedelta(seconds=end-start)}.\n')

# Save the numpy array form of the dataset in order to validate
# correctness of decompression
np.save('data_out', decomp_data)

full_end = timer()
print(f'TOTAL TIME: {timedelta(seconds=full_end-full_start)}.\n')
