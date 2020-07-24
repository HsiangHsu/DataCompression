#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from drivers import decode

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='comp.out',
    help='compressed data to decompress')
parser.add_argument('-a', type=str, default='args.out',
    help='arguments used to compress data')
parser.add_argument('-p', type=str, required=False,
    help='name to generate from')

args = parser.parse_args()
if args.p:
    args.c = f'results/{args.p}_comp.out'
    args.a = f'results/{args.p}_args.out'

with open(args.a, 'rb') as f:
    comp_args = pickle.load(f)

assert comp_args.enc == 'pred-huff'

compression, pre_metadata, comp_metadata, original_shape = decode(args.c,
    comp_args)
error_string, residuals, clf = compression

plt.hist(residuals.flatten().astype(np.uint8), bins=256)
plt.savefig(f'results/{args.p}_residuals.png')
