'''
encoders/predictive_huff_run.py

This module contains the Huffman encoder for use with predictive coding.
'''

from bitstring import BitArray
from copy import deepcopy
from golomb_coding import golomb_coding
from heapq import heappush, heappop, heapify
from humanize import naturalsize
from itertools import chain, groupby, tee
from math import ceil, log2
from minimal_binary_coding import minimal_binary_coding
import numpy as np
import pickle
import re

from utilities import readint, encode_predictor, decode_predictor, \
    write_shape, read_shape

error_k = 32
residual_k = 32


def pred_huff_run_enc(compression, pre_metadata, original_shape, args):
    '''
    Huffman encoder for predictive coding.

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

    error_shape = error_string.shape
    error_string = error_string.flatten()
    residual_shape = residuals.shape
    residuals = residuals.flatten()

    # Generate Huffman code for error string
    freqs = get_freqs(error_string)
    raw_code = huffman_encode(freqs)
    n_error_symbols = len(raw_code)
    error_code = dict(raw_code)
    symbol_len = error_string.dtype.itemsize
    max_code_len = len(max(raw_code, key=lambda c: len(c[1]))[1])
    error_code_len = ceil(max_code_len/8)
    error_codestream = b''
    for symbol in raw_code:
        error_codestream += int(symbol[0]).to_bytes(symbol_len, 'little')
        error_codestream += len(symbol[1]).to_bytes(error_code_len, 'little')

    # Generate Huffman code for residuals
    freqs = get_freqs(residuals)
    raw_code = huffman_encode(freqs)
    n_residual_symbols = len(raw_code)
    residual_code = dict(raw_code)
    max_code_len = len(max(raw_code, key=lambda c: len(c[1]))[1])
    residual_code_len = ceil(max_code_len/8)
    residual_codestream = b''
    for symbol in raw_code:
        residual_codestream += int(symbol[0]).to_bytes(symbol_len, 'little')
        residual_codestream += len(symbol[1]).to_bytes(residual_code_len,
            'little')

    codestream = b''
    codestream += write_shape(error_shape)
    codestream += write_shape(residual_shape)
    codestream += symbol_len.to_bytes(1, 'little')
    codestream += error_code_len.to_bytes(1, 'little')
    codestream += residual_code_len.to_bytes(1, 'little')
    codestream += len(error_codestream).to_bytes(4, 'little')
    codestream += len(residual_codestream).to_bytes(4, 'little')
    codestream += error_codestream
    codestream += residual_codestream
    f.write(codestream)
    codelen = len(codestream)
    print(f'\tCodestream: {naturalsize(codelen)}, ' + \
        f'{n_error_symbols} / {n_residual_symbols} symbols.')

    # Generate error bytestreams
    error_syms, error_rls = to_run_length(error_string)

    error_sym_bitstream = ''
    for error_sym in error_syms:
        error_sym_bitstream += error_code[error_sym]
    error_sym_padded_length = 8*ceil(len(error_sym_bitstream)/8)
    error_sym_padding = error_sym_padded_length - \
        len(error_sym_bitstream);
    error_sym_bitstream = \
        f'{error_sym_bitstream:0<{error_sym_padded_length}}'
    error_sym_bytestream = [int(error_sym_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_sym_bitstream)//8)]

    error_rl_bitstream = golomb_encode(error_rls, error_k)
    error_rl_padded_length = 8*ceil(len(error_rl_bitstream)/8)
    error_rl_padding = error_rl_padded_length - len(error_rl_bitstream);
    error_rl_bitstream = f'{error_rl_bitstream:0<{error_rl_padded_length}}'
    error_rl_bytestream = [int(error_rl_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_rl_bitstream)//8)]

    # Generate residual bytestreams
    residual_syms, residual_rls = to_run_length(residuals)

    residual_sym_bitstream = ''
    for residual_sym in residual_syms:
        residual_sym_bitstream += residual_code[residual_sym]
    residual_sym_padded_length = 8*ceil(len(residual_sym_bitstream)/8)
    residual_sym_padding = residual_sym_padded_length - \
        len(residual_sym_bitstream);
    residual_sym_bitstream = \
        f'{residual_sym_bitstream:0<{residual_sym_padded_length}}'
    residual_sym_bytestream = [int(residual_sym_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(residual_sym_bitstream)//8)]

    residual_rl_bitstream = golomb_encode(residual_rls, residual_k)
    residual_rl_padded_length = 8*ceil(len(residual_rl_bitstream)/8)
    residual_rl_padding = residual_rl_padded_length - \
        len(residual_rl_bitstream);
    residual_rl_bitstream = \
        f'{residual_rl_bitstream:0<{residual_rl_padded_length}}'
    residual_rl_bytestream = [int(residual_rl_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(residual_rl_bitstream)//8)]

    bytestream = b''
    bytestream += error_sym_padding.to_bytes(1, 'little')
    bytestream += error_rl_padding.to_bytes(1, 'little')
    bytestream += residual_sym_padding.to_bytes(1, 'little')
    bytestream += residual_rl_padding.to_bytes(1, 'little')

    bytestream += len(error_sym_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_sym_bytestream)

    bytestream += len(error_rl_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_rl_bytestream)

    bytestream += len(residual_sym_bytestream).to_bytes(4, 'little')
    bytestream += bytes(residual_sym_bytestream)

    bytestream += len(residual_rl_bytestream).to_bytes(4, 'little')
    bytestream += bytes(residual_rl_bytestream)
    f.write(bytestream)
    bytelen = len(bytestream)
    print(f'\tBytestream: {naturalsize(bytelen)}.')

    print(f'\tTotal len: {naturalsize(metalen+codelen+bytelen)}.\n')

    f.close()

    with open('args.out', 'wb') as f:
        pickle.dump(args, f)


def pred_huff_run_dec(comp_file):
    '''
    Delta Vector and Huffman Encoding Decoder

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

    # Reconstruct Huffman code
    error_shape = read_shape(f)
    residual_shape = read_shape(f)
    symbol_len = readint(f, 1)
    error_code_len = readint(f, 1)
    residual_code_len = readint(f, 1)
    error_codestream_len = readint(f, 4)
    residual_codestream_len = readint(f, 4)

    # Error codestream
    codebytes_read = 0
    raw_code = []
    while codebytes_read < error_codestream_len:
        raw_code.append([readint(f, symbol_len), readint(f, error_code_len)])
        codebytes_read += symbol_len + error_code_len
    error_code = deepcopy(raw_code)
    codelen = raw_code[0][1]
    code = f'{0:0{codelen}b}'
    error_code[0][1] = code
    for k in range(1, len(raw_code)):
        codelen = raw_code[k][1]
        code = int(code, 2) + 1
        code = code << (codelen - raw_code[k-1][1])
        code = f'{code:0{codelen}b}'
        error_code[k][1] = code

    # Residual codestream
    codebytes_read = 0
    raw_code = []
    while codebytes_read < residual_codestream_len:
        raw_code.append([readint(f, symbol_len),
            readint(f, residual_code_len)])
        codebytes_read += symbol_len + residual_code_len
    residual_code = deepcopy(raw_code)
    codelen = raw_code[0][1]
    code = f'{0:0{codelen}b}'
    residual_code[0][1] = code
    for k in range(1, len(raw_code)):
        codelen = raw_code[k][1]
        code = int(code, 2) + 1
        code = code << (codelen - raw_code[k-1][1])
        code = f'{code:0{codelen}b}'
        residual_code[k][1] = code

    error_sym_padding_bits = readint(f, 1)
    error_rl_padding_bits = readint(f, 1)
    residual_sym_padding_bits = readint(f, 1)
    residual_rl_padding_bits = readint(f, 1)

    error_sym_bytestream_len = readint(f, 4)
    error_sym_bitstream_len = error_sym_bytestream_len*8 - \
        error_sym_padding_bits
    error_sym_bytestream = f.read(error_sym_bytestream_len)

    error_rl_bytestream_len = readint(f, 4)
    error_rl_bitstream_len = error_rl_bytestream_len*8 - \
        error_rl_padding_bits
    error_rl_bytestream = f.read(error_rl_bytestream_len)

    residual_sym_bytestream_len = readint(f, 4)
    residual_sym_bitstream_len = residual_sym_bytestream_len*8 - \
        residual_sym_padding_bits
    residual_sym_bytestream = f.read(residual_sym_bytestream_len)

    residual_rl_bytestream_len = readint(f, 4)
    residual_rl_bitstream_len = residual_rl_bytestream_len*8 - \
        residual_rl_padding_bits
    residual_rl_bytestream = f.read(residual_rl_bytestream_len)

    error_decodings = {v: k for k, v in dict(error_code).items()}
    residual_decodings = {v: k for k, v in dict(residual_code).items()}
    error_syms = huffman_decode(error_sym_bytestream,
        error_sym_bitstream_len, error_decodings)
    residual_syms = huffman_decode(residual_sym_bytestream,
        residual_sym_bitstream_len, residual_decodings)
    error_rls = golomb_decode(error_rl_bytestream, error_rl_bitstream_len,
        error_k)
    residual_rls = golomb_decode(residual_rl_bytestream,
        residual_rl_bitstream_len, residual_k)

    error_stream = from_run_length(error_syms, error_rls)
    residual_stream = from_run_length(residual_syms, residual_rls)

    error_string = np.array(error_stream, dtype=dtype).reshape(error_shape)
    residuals = np.array(residual_stream, dtype=dtype).reshape(residual_shape)

    return (error_string, residuals, clf), None, (n_prev, pcs, ccs), \
        original_shape


def huffman_encode(symb2freq):
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    non_canonical = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    canonical = deepcopy(non_canonical)
    codelen = len(non_canonical[0][1])
    code = f'{0:0{codelen}b}'
    canonical[0][1] = code
    for i in range(1, len(non_canonical)):
        codelen = len(non_canonical[i][1])
        code = int(code, 2) + 1
        code = code << (codelen - len(non_canonical[i-1][1]))
        code = f'{code:0{codelen}b}'
        canonical[i][1] = code

    return canonical


def huffman_decode(bytestream, length, decodings):
    bitstream_array = BitArray(bytestream).bin
    decoded_stream = list()
    index = 0
    while (index < length):
        possible_encoding = bitstream_array[index]
        possible_end_index = index + 1
        while (possible_encoding not in decodings):
            possible_end_index += 1
            possible_encoding = bitstream_array[index:possible_end_index]
        decoded_stream.append(decodings[possible_encoding])
        index = possible_end_index
    return decoded_stream


def get_freqs(values):
    freqs = dict()
    for val in values:
        try:
            freqs[val] += 1
        except KeyError:
            freqs[val] = 1
    return freqs


def to_run_length(stream):
    symbols, run_lengths = zip(*[(a, len([*b])) for a, b in groupby(stream)])
    return symbols, run_lengths


def from_run_length(symbols, run_lengths):
    return list(chain.from_iterable(
        [[a for i in range(b)] for a, b in list(zip(symbols, run_lengths))]))


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
