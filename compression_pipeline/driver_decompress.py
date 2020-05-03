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

import decompressor
import postprocessor

parser = argparse.ArgumentParser()
parser.add_argument('data_in', type=str, help='compressed data to decompress')
parser.add_argument('args_in', type=str,
    help='arguments used to compress data')

args = parser.parse_args()

with open(args.args_in, 'rb') as f:
    comp_args = pickle.load(f)

compression, metadata = decompressor.decode(args.data_in, comp_args['enc'])
decompressed_data = decompressor.decompress(compression, metadata,
    comp_args['comp'])
postprocessed_data = postprocessor.postprocess(decompressed_data,
    comp_args['pre'])

# Save the numpy array form of the dataset in order to validate
# correctness of decompression
np.save('data_out', postprocessed_data)
