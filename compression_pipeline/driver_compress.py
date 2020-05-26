'''
driver_compress.py

This is the executable used to compress a dataset.
All parameters for compression are passed in as command-line arguments.
After parsing arguments, the work is delegated to the loader,
preprocessor, and compressor modules.
'''


import argparse
import numpy as np
import sys

import loader
import preprocessor
import compressor

from datetime import timedelta
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset to compress',
    choices = ['test', 'mnist', 'cifar-10', 'synthetic'])

pre_group = parser.add_argument_group('preprocessor')
pre_group.add_argument('--pre', type=str, choices=['sqpatch'],
    help='preprocessor to use', dest='pre')
pre_group.add_argument('--psz', type=int,
    help='dimension of cropped patch for sqpatch',
    required='sqpatch' in sys.argv, dest='psz')

comp_group = parser.add_argument_group('compressor')
comp_group.add_argument('--comp', type=str, choices=['knn-mst'],
    help='compressor to use', dest='comp', required=True)
comp_group.add_argument('--nneigh', type=int,
    help='number of neighbors for knn-mst',
    required='knn-mst' in sys.argv, dest='n_neighbors')
comp_group.add_argument('--metric', type=str,
    help='distance metric for knn-mst', choices=['hamming', 'minkowski'],
    required='knn-mst' in sys.argv, dest='metric')
comp_group.add_argument('--minkp', type=int,
    help='parameter for Minkowski metric', dest='minkowski_p', default=2)
comp_group.add_argument('--enc', type=str,
    choices=['delta-coo', 'delta-huff'],
    help='encoder to use', dest='enc', required=True)

args = parser.parse_args()

full_start = timer()

start = timer()
data, labels = loader.load_dataset(args.dataset)
end = timer()
print(f'load in {timedelta(seconds=end-start)}.\n')

# Save the numpy array form of the dataset in order to validate
# correctness of decompression
np.save('data_in', data)

start = timer()
if args.pre:
    data, element_axis = preprocessor.preprocess(data, args.pre, psz=args.psz)
else:
    element_axis = 0
end = timer()
print(f'preprocess in {timedelta(seconds=end-start)}.\n')

start = timer()
compressed_data, local_metadata, original_shape = compressor.compress(data,
    element_axis, args.comp, n_neighbors=args.n_neighbors, metric=args.metric,
    minkowski_p=args.minkowski_p)
end = timer()
print(f'compress in {timedelta(seconds=end-start)}.\n')

start = timer()
compressor.encode(compressed_data, local_metadata, original_shape, args.enc,
    vars(args))
end = timer()
print(f'encode in {timedelta(seconds=end-start)}.\n')

full_end = timer()
print(f'TOTAL TIME: {timedelta(seconds=full_end-full_start)}.\n')
