#!/usr/bin/env python3

'''
hist.py

Script to plot a histogram of error string, residuals, etc. for predictive
coding.

NB: Current code below is for plotting frequency of symbols in the error string
except for the 5 most frequent.
'''


import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from drivers import decode
from utilities import get_freqs

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='comp.out',
    help='compressed data to decompress')
parser.add_argument('-a', type=str, default='args.out',
    help='arguments used to compress data')
parser.add_argument('-p', type=str, required=False,
    help='name to generate from')

args = parser.parse_args()
if args.p:
    args.c = f'{args.p}_comp.out'
    args.a = f'{args.p}_args.out'

with open(args.a, 'rb') as f:
    comp_args = pickle.load(f)

compression, pre_metadata, comp_metadata, original_shape = decode(args.c,
    comp_args)
error_string, residuals, clf = compression
freqs = get_freqs(error_string.flatten())
freqs = [(item[0], item[1]) for item in freqs.items()]
freqs.sort(key= lambda x: -x[1])

to_chop = 5
x = [item[0] for item in freqs[to_chop:]]
y = [item[1] for item in freqs[to_chop:]]
plt.scatter(x, y)
plt.savefig('linear.png')
