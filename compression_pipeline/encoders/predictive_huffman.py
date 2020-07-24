'''
encoders/predictive_huffman.py

This module contains the Huffman encoder for use with predictive coding.
'''

from bitstring import BitArray
from copy import deepcopy
from heapq import heappush, heappop, heapify
from humanize import naturalsize
from math import ceil, log2
import numpy as np
import pickle

from utilities import readint, encode_predictor, decode_predictor, \
    write_shape, read_shape


def pred_huffman_enc(compression, pre_metadata, original_shape, args):
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
    # f = open(f'data/predictive/{args.dataset}_{args.predictor_family}_{args.prev_context}_{args.curr_context}_{args.num_prev_imgs}_comp.out', 'wb')
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

    # Generate Huffman coded bitstream
    error_bitstream = ''
    for error in error_string:
        error_bitstream += error_code[error]
    error_padded_length = 8*ceil(len(error_bitstream)/8)
    error_padding = error_padded_length - len(error_bitstream);
    error_bitstream = f'{error_bitstream:0<{error_padded_length}}'
    error_bytestream = [int(error_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_bitstream)//8)]

    residual_bitstream = ''
    for residual in residuals:
        residual_bitstream += residual_code[residual]
    residual_padded_length = 8*ceil(len(residual_bitstream)/8)
    residual_padding = residual_padded_length - len(residual_bitstream);
    residual_bitstream = f'{residual_bitstream:0<{residual_padded_length}}'
    residual_bytestream = [int(residual_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(residual_bitstream)//8)]

    bytestream = b''
    bytestream += error_padding.to_bytes(1, 'little')
    bytestream += residual_padding.to_bytes(1, 'little')
    bytestream += len(error_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_bytestream)
    bytestream += len(residual_bytestream).to_bytes(4, 'little')
    bytestream += bytes(residual_bytestream)
    f.write(bytestream)
    bytelen = len(bytestream)
    print(f'\tBytestream: {naturalsize(bytelen)}.')

    print(f'\tTotal len: {naturalsize(metalen+codelen+bytelen)}.\n')

    f.close()

    with open('args.out', 'wb') as f:
    # with open(f'data/predictive/{args.dataset}_{args.predictor_family}_{args.prev_context}_{args.curr_context}_{args.num_prev_imgs}_args.out', 'wb') as f:
        pickle.dump(args, f)


def pred_huffman_dec(comp_file):
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

    error_padding_bits = readint(f, 1)
    residual_padding_bits = readint(f, 1)
    error_bytestream_len = readint(f, 4)
    error_bitstream_len = error_bytestream_len*8 - error_padding_bits
    error_bytestream = f.read(error_bytestream_len)
    residual_bytestream_len = readint(f, 4)
    residual_bytestream = f.read(residual_bytestream_len)
    residual_bitstream_len = residual_bytestream_len*8 - residual_padding_bits

    error_string = np.empty((n_errors,), dtype=dtype)
    residuals = np.empty((n_residuals,), dtype=dtype)

    error_decodings = {v: k for k, v in dict(error_code).items()}
    residual_decodings = {v: k for k, v in dict(residual_code).items()}
    error_decoded_stream = huffman_decode(error_bytestream,
        error_bitstream_len, error_decodings)
    residual_decoded_stream = huffman_decode(residual_bytestream,
        residual_bitstream_len, residual_decodings)

    error_string = np.array(error_decoded_stream, dtype=dtype).reshape(
        error_shape)
    residuals = np.array(residual_decoded_stream, dtype=dtype).reshape(
        residual_shape)

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
