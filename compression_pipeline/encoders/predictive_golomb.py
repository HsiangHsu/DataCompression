'''
encoders/predictive_golomb.py

This module contains the Golomb encoder for use with predictive coding.
'''

from bitstring import BitArray
from copy import deepcopy
from golomb_coding import golomb_coding
from heapq import heappush, heappop, heapify
from humanize import naturalsize
from itertools import tee
from math import ceil, log2
from minimal_binary_coding import minimal_binary_coding
import numpy as np
import pickle
import re

from utilities import readint, encode_predictor, decode_predictor, \
    write_shape, read_shape

from datetime import timedelta
from timeit import default_timer as timer


def pred_golomb_enc(compression, pre_metadata, original_shape, args):
    '''
    Golomb encoder for predictive coding.

    Args:
    compression: (numpy array, numpy array, list)
        compression as returned by the predictive preprocessor, with an
        error string, a residual string, and a list of predictors
    pre_metadata: (int, string, string)
        metadata as returned by the predictive preprocessor, with the number of
        previous images, the previous context string, and the current context
        string
    original_shape: tuple
        shape of original data
    args: Namespace
        command-line argument namespace

    Returns:
        None
    '''

    error_string, residuals, clf = compression
    n_clf = len(clf)
    n_errors = error_string.shape[0]
    n_residuals = residuals.shape[0]
    n_prev = pre_metadata[0]
    pcs = pre_metadata[1]
    ccs = pre_metadata[2]

    # Bytestreams to be built and written
    metastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
    metastream += n_clf.to_bytes(1, 'little')
    metastream += n_errors.to_bytes(4, 'little')
    metastream += n_residuals.to_bytes(4, 'little')
    metastream += ord(error_string.dtype.char).to_bytes(1, 'little')
    metastream += write_shape(original_shape)
    metastream += encode_predictor(clf)
    metastream += n_prev.to_bytes(1, 'little')
    metastream += len(pcs).to_bytes(1, 'little')
    metastream += pcs.encode()
    metastream += len(ccs).to_bytes(1, 'little')
    metastream += ccs.encode()

    f = open('comp.out', 'wb')
    metalen = len(metastream)
    print(f'\tMetastream: {naturalsize(metalen)}.')
    f.write(metastream)

    np.save('err_in', error_string)

    # Generate Golomb coded bytestream
    error_shape = error_string.shape
    error_string = error_string.flatten()
    residual_shape = residuals.shape
    residuals = residuals.flatten()

    error_bitstream = golomb_encode(error_string, args.error_k)
    padded_length = 8*ceil(len(error_bitstream)/8)
    error_padding = padded_length - len(error_bitstream);
    error_bitstream = f'{error_bitstream:0<{padded_length}}'
    error_bytestream = [int(error_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_bitstream)//8)]

    residual_bitstream = golomb_encode(residuals, args.residual_k)
    padded_length = 8*ceil(len(residual_bitstream)/8)
    residual_padding = padded_length - len(residual_bitstream);
    residual_bitstream = f'{residual_bitstream:0<{padded_length}}'
    residual_bytestream = [int(residual_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(residual_bitstream)//8)]

    bytestream = b''
    bytestream += write_shape(error_shape)
    bytestream += write_shape(residual_shape)
    bytestream += error_padding.to_bytes(1, 'little')
    bytestream += residual_padding.to_bytes(1, 'little')
    bytestream += args.error_k.to_bytes(1, 'little')
    bytestream += args.residual_k.to_bytes(1, 'little')
    bytestream += len(error_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_bytestream)
    bytestream += len(residual_bytestream).to_bytes(4, 'little')
    bytestream += bytes(residual_bytestream)
    f.write(bytestream)
    bytelen = len(bytestream)
    print(f'\tBytestream: {naturalsize(bytelen)}.')

    print(f'\tTotal len: {naturalsize(metalen+bytelen)}.\n')

    f.close()

    with open('args.out', 'wb') as f:
        pickle.dump(args, f)


def pred_golomb_dec(comp_file):
    '''
    Golomb decoder

    See docstring on the corresponding encoder for more information.

    Args:
        comp_file: string
            path to encoded compression

    Returns:
        compression: numpy array
            compressed data
        pre_metadata: numpy array
            metadata for preprocessing
        comp_metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
    '''

    f = open(comp_file, 'rb')

    # Read in sizing and and datatype metadata to reconstruct arrays.
    n_pred = readint(f, 1)
    n_errors = readint(f, 4)
    n_residuals = readint(f, 4)
    dtype = np.dtype(chr(readint(f, 1)))
    dsize = dtype.itemsize
    original_shape = read_shape(f)
    clf = decode_predictor(f, n_pred)
    n_prev = readint(f, 1)
    len_pcs = readint(f, 1)
    pcs = f.read(len_pcs).decode()
    len_ccs = readint(f, 1)
    ccs = f.read(len_ccs).decode()

    error_shape = read_shape(f)
    residual_shape = read_shape(f)
    error_padding_bits = readint(f, 1)
    residual_padding_bits = readint(f, 1)
    error_k = readint(f, 1)
    residual_k = readint(f, 1)
    error_bytestream_len = readint(f, 4)
    error_bitstream_len = error_bytestream_len*8 - error_padding_bits
    error_bytestream = f.read(error_bytestream_len)
    residual_bytestream_len = readint(f, 4)
    residual_bytestream = f.read(residual_bytestream_len)
    residual_bitstream_len = residual_bytestream_len*8 - residual_padding_bits

    error_string = np.empty((n_errors,), dtype=dtype)
    residuals = np.empty((n_residuals,), dtype=dtype)

    error_decoded_stream = golomb_decode(error_bytestream, error_bitstream_len,
        error_k)
    residual_decoded_stream = golomb_decode(residual_bytestream,
        residual_bitstream_len, residual_k)

    error_string = np.array(error_decoded_stream, dtype=dtype).reshape(
        error_shape)
    residuals = np.array(residual_decoded_stream, dtype=dtype).reshape(
        residual_shape)

    np.save('err_out', error_string)

    return (error_string, residuals, clf), (n_prev, pcs, ccs), \
        (n_prev, pcs, ccs), original_shape


def golomb_encode(stream, k):
    encoded_stream = ['0'*(val//k)+'1'+minimal_binary_coding(val%k,k)
        for val in stream]
    return ''.join(encoded_stream)


def golomb_decode(bytestream, bitstream_len, k):
    bitstream = BitArray(bytestream).bin
    bits_read = 0
    base = int(log2(k))

    idxs = [m.end() for m in re.finditer(f'1.{{{base}}}', bitstream)]
    idxs.insert(0, 0)
    start, end = tee(idxs)
    next(end, None)

    words = [bitstream[i:j] for i, j in zip(start, end)]
    parts = [word.partition('1') for word in words]
    decoded_stream = [(len(u)<<base) + int(b, base=2) for u, _, b in parts]

    return decoded_stream
