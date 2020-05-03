import argparse
import numpy as np
import pickle
import sys

import decompressor

parser = argparse.ArgumentParser()
parser.add_argument('data_in', type=str, help='compressed data to decompress')
parser.add_argument('args_in', type=str,
    help='arguments used to compress data')

args = parser.parse_args()

with open(args.args_in, 'rb') as f:
    comp_args = pickle.load(f)

decompressed_data = decompressor.decompress(args.data_in, comp_args)
np.save('data_out', decompressed_data)
