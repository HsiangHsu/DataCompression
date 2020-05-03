import argparse
import numpy as np
import sys

import loader
import preprocessor
import compressor

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset to compress',
    choices = ['test', 'mnist', 'cifar-10'])

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
comp_group.add_argument('--enc', type=str, choices=['delta-coo'],
    help='encoder to use', dest='enc', required=True)

args = parser.parse_args()

data, labels = loader.load_dataset(args.dataset)
np.save('data_in', data)
preprocessed_data = preprocessor.preprocess(data, args.pre, psz=args.psz)
compressed_data, metadata = compressor.compress(preprocessed_data, args.comp,
    n_neighbors=args.n_neighbors, metric=args.metric,
    minkowski_p=args.minkowski_p)
compressor.encode(compressed_data, metadata, args.enc, args)
