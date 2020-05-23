import argparse
import gzip
import os
import shutil
import tarfile

import loader

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset to benchmark',
    choices = ['test', 'mnist', 'cifar-10', 'synthetic'])
args = parser.parse_args()

paths_in = loader.get_dataset_path(args.dataset)

# GZIP
with tarfile.open(args.dataset+'.tar.gz', 'x:gz') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))

# BZIP2
with tarfile.open(args.dataset+'.tar.bz2', 'x:bz2') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))

# LZMA
with tarfile.open(args.dataset+'.tar.xz', 'x:xz') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))
