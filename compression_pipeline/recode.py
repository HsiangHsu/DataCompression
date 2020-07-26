#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pickle

from drivers import decode, encode

from datetime import timedelta
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--comp', type=str, default='comp.out',
    help='compressed data to decompress')
parser.add_argument('--args', type=str, default='args.out',
    help='arguments used to compress data')
parser.add_argument('--enc', type=str,
    choices=['delta-coo', 'delta-huff', 'video', 'pred-huff', 'pred-golomb'],
    help='encoder to use', dest='enc', default='delta-huff')
parser.add_argument('--error-k', type=int,
    help='golomb paramater for error string', dest='error_k',
    default=8)
parser.add_argument('--residual-k', type=int,
    help='golomb paramater for error string', dest='residual_k',
    default=32)

args = parser.parse_args()

full_start = timer()

with open(args.args, 'rb') as f:
    comp_args = pickle.load(f)

if comp_args.enc in ['pred-huff', 'pred-golomb']:
    assert args.enc in ['pred-huff', 'pred-golomb'], \
        'A predictive encoding scheme is required.'

start = timer()
compression, pre_metadata, comp_metadata, original_shape = decode(args.comp,
    comp_args)
end = timer()
print(f'\nORIGINAL: {comp_args.enc}')
print(f'decode in {timedelta(seconds=end-start)}.\n')

comp_args.enc = args.enc

start = timer()
print(f'NEW: {comp_args.enc}\n')
encode(compression, pre_metadata, comp_metadata, original_shape, comp_args)
end = timer()
print(f'encode in {timedelta(seconds=end-start)}.\n')

full_end = timer()
print(f'TOTAL TIME: {timedelta(seconds=full_end-full_start)}.\n')
