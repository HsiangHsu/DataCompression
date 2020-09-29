import argparse
import gzip
import os
import shutil
import tarfile

import loader

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, help='dataset to benchmark',
    choices=['test', 'mnist', 'cifar-10', 'synthetic', 'adult', 'utk-face'])
parser.add_argument('-f', type=str, help='file to benchmark')
args = parser.parse_args()

if args.d and args.f:
    parser.error('must supply either a dataset or file, not both')

if args.d:
    paths_in = loader.get_dataset_path(args.d)
    filename = args.d
elif args.f:
    paths_in = [args.f]
    filename = os.path.splitext(os.path.split(args.f)[-1])[0]
else:
    parser.error('must supply a dataset or file')

# GZIP
with tarfile.open(filename+'.tar.gz', 'x:gz') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))

# BZIP2
with tarfile.open(filename+'.tar.bz2', 'x:bz2') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))

# LZMA
with tarfile.open(filename+'.tar.xz', 'x:xz') as tar:
    for path_in in paths_in:
        tar.add(path_in, os.path.basename(path_in))
