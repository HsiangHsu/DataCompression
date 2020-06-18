#!/usr/bin/env python3

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

from drivers import preprocess, compress, encode
from loader import load

from datetime import timedelta
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset to compress',
    choices = ['test', 'mnist', 'cifar-10', 'adult', 'synthetic'])

pre_group = parser.add_argument_group('preprocessor')
pre_group.add_argument('--pre', type=str,
    choices=['sqpatch', 'rgb', 'rgb-sqpatch', 'dct'],
    help='preprocessor to use', dest='pre')
pre_group.add_argument('--rgb-r', type=int,
    help='rows in rgb data', dest='rgbr')
pre_group.add_argument('--rgb-c', type=int,
    help='cols of rgb data', dest='rgbc')
pre_group.add_argument('--psz', type=int,
    help='dimension of cropped patch for sqpatch')

comp_group = parser.add_argument_group('compressor')
comp_group.add_argument('--comp', type=str, choices=['knn-mst'],
    help='compressor to use', dest='comp', required=True)
comp_group.add_argument('--metric', type=str,
    help='distance metric for knn-mst', choices=['hamming', 'minkowski'],
    required='knn-mst' in sys.argv, dest='metric')
comp_group.add_argument('--minkp', type=int,
    help='parameter for Minkowski metric', dest='minkowski_p', default=2)
comp_group.add_argument('--enc', type=str,
    choices=['delta-coo', 'delta-huff', 'video'],
    help='encoder to use', dest='enc', required=True)

video_enc_group = parser.add_argument_group('video encoding')
valid_intermediate_frame_codecs = ['jpg', 'png']
valid_output_video_codecs = ['av1', 'vp8', 'vp9']
video_enc_group.add_argument(
        '--video_codec', default='vp8',
        help='video codec to be used for output (av1, vp8, vp9)')
video_enc_group.add_argument(
        '--image_codec', default='png',
        help='intermediate image frame codec (png, jpg)')
video_enc_group.add_argument(
        '--framerate', default=24, type=int)
# TODO(mbarowsky)
# Add better GoP (default, max, longest_path/some heuristic) argument
video_enc_group.add_argument(
        '--gop_strat', default='default',
        help='how GoP value should be determined (default, max, [INT])')

args = parser.parse_args()

# Argument validation
if args.enc == 'video' and not args.comp == 'knn-mst':
    parser.error('video encoding expects use of KNN-MST ordering')
if args.image_codec not in valid_intermediate_frame_codecs:
    parser.error('intermediate frame codec must be png or jpg')
if args.video_codec not in valid_output_video_codecs:
    parser.error('output video codec must be av1, vp8, or vp9')
if args.gop_strat not in ['default', 'max']:
    try:
        int(args.gop_strat)
    except ValueError:
        parser.error('GoP strategy must be default, max, or an integer')
if args.framerate < 1:
    parser.error('framerate must be >= 1')

if (args.pre == 'sqpatch' or args.pre == 'rgb-sqpatch') and not args.psz:
    parser.error('must supply --psz for sqpatch')
if (args.pre == 'rgb' or args.pre == 'rgb-sqpatch') and \
    (not (args.rgbr and args.rgbc)):
    parser.error('must supply --rgb-r and --rgb-c for rgb')


full_start = timer()

start = timer()
data, labels = load(args.dataset)
end = timer()
print(f'\nload in {timedelta(seconds=end-start)}.\n')

# Save the numpy array form of the dataset in order to validate
# correctness of decompression
np.save('data_in', data)

start = timer()
if args.pre:
    data, element_axis = preprocess(data, args)
else:
    element_axis = 0
end = timer()
print(f'preprocess in {timedelta(seconds=end-start)}.\n')

start = timer()
compressed_data, local_metadata, original_shape = compress(data, element_axis,
    args)
end = timer()
print(f'compress in {timedelta(seconds=end-start)}.\n')

if args.video_codec:
    if args.dataset == 'mnist':
        args.grayscale = True
    else:
        args.grayscale = False

start = timer()
encode(compressed_data, local_metadata, original_shape, args)
end = timer()
print(f'encode in {timedelta(seconds=end-start)}.\n')

full_end = timer()
print(f'TOTAL TIME: {timedelta(seconds=full_end-full_start)}.\n')
