'''
encoders/predictive_huff_run.py

This module contains the Huffman encoder for use with predictive coding.
'''

from copy import deepcopy
from humanize import naturalsize
from math import ceil, log2
import numpy as np
import pickle

from utilities import readint, encode_predictor, decode_predictor, \
    write_shape, read_shape, get_freqs, huffman_encode, huffman_decode, \
    golomb_encode, golomb_decode, to_run_length, from_run_length


def pred_huff_run_enc(compression, pre_metadata, original_shape, args):
    '''
    Predictive Huffman with Golomb Run-Length Encoder

    Encoder for the predictive coding compressor that encodes the error
    string and residuals using Huffman codes, but run-length encodes the
    Huffman encoding for the error string using a Golomb code.
    Separate Huffman codes are used for the error string and residuals.

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

    f = open('comp.out', 'wb')

    # Unpack arguments
    error_string, residuals, clf = compression
    n_clf = len(clf)
    n_errors = error_string.shape[0]
    n_residuals = residuals.shape[0]
    n_prev = pre_metadata[0]
    pcs = pre_metadata[1]
    ccs = pre_metadata[2]

    # metastream contains data necessary for the decoder to recreate
    # objects such as numpy arrays and sklearn models
    metastream = b''
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
    f.write(metastream)

    metalen = len(metastream)
    print(f'\tMetastream: {naturalsize(metalen)}.')

    # Generate Huffman code for error string
    # n_error_symbols: number of symbols in codebook
    # symbol_len: bytes needed for uncoded symbol
    # error_code_len: bytes needed for coded symbol
    error_shape = error_string.shape
    error_string = error_string.flatten()
    freqs = get_freqs(error_string)
    raw_code = huffman_encode(freqs)
    n_error_symbols = len(raw_code)
    error_code = dict(raw_code)
    symbol_len = error_string.dtype.itemsize
    max_code_len = len(max(raw_code, key=lambda c: len(c[1]))[1])
    error_code_len = ceil(max_code_len/8)

    # Write the codebook for the error string
    error_codestream = b''
    for symbol in raw_code:
        error_codestream += int(symbol[0]).to_bytes(symbol_len, 'little')
        error_codestream += len(symbol[1]).to_bytes(error_code_len, 'little')

    # Generate Huffman code for residuals
    residual_shape = residuals.shape
    residuals = residuals.flatten()
    freqs = get_freqs(residuals)
    raw_code = huffman_encode(freqs)
    n_residual_symbols = len(raw_code)
    residual_code = dict(raw_code)
    max_code_len = len(max(raw_code, key=lambda c: len(c[1]))[1])
    residual_code_len = ceil(max_code_len/8)

    # Write the codebook for the residuals
    residual_codestream = b''
    for symbol in raw_code:
        residual_codestream += int(symbol[0]).to_bytes(symbol_len, 'little')
        residual_codestream += len(symbol[1]).to_bytes(residual_code_len,
            'little')

    # codestream contains the data for reconstructing the Huffman codebooks
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

    # Calculate data for run-length encoding
    # error_syms is an array of symbols in the order they appear
    # error_rls is an array of ints corresponding to how many times the symbol
    # at the corresponding index in error_syms appears
    # Ex: [5, 5, 5, 4, 4, 1, 4] yields:
    #   error_syms: (5, 4, 1, 4)
    #   error_rls: (3, 2, 1, 1)
    error_syms, error_rls = to_run_length(error_string)

    # Generate Huffman encoding for the error string symbols
    # error_sym__bitstream is a string of 0s and 1s
    # error_sym__bytestream is a padded bytestring that can be written to disk
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

    # Generate Golomb encoding for the error string run lengths
    error_rl_bitstream = golomb_encode(error_rls, args.error_k)
    error_rl_padded_length = 8*ceil(len(error_rl_bitstream)/8)
    error_rl_padding = error_rl_padded_length - len(error_rl_bitstream);
    error_rl_bitstream = f'{error_rl_bitstream:0<{error_rl_padded_length}}'
    error_rl_bytestream = [int(error_rl_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_rl_bitstream)//8)]

    # Generate Huffman encoding for the residuals
    # Residuals are not run-length encoded
    residual_bitstream = ''
    for residual in residuals:
        residual_bitstream += residual_code[residual]
    residual_padded_length = 8*ceil(len(residual_bitstream)/8)
    residual_padding = residual_padded_length - len(residual_bitstream);
    residual_bitstream = f'{residual_bitstream:0<{residual_padded_length}}'
    residual_bytestream = [int(residual_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(residual_bitstream)//8)]

    # bytestream contains the actual encodings and is the largest component
    # of the final compression size
    bytestream = b''
    bytestream += error_sym_padding.to_bytes(1, 'little')
    bytestream += error_rl_padding.to_bytes(1, 'little')
    bytestream += residual_padding.to_bytes(1, 'little')
    bytestream += args.error_k.to_bytes(1, 'little')
    bytestream += len(error_sym_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_sym_bytestream)
    bytestream += len(error_rl_bytestream).to_bytes(4, 'little')
    bytestream += bytes(error_rl_bytestream)
    bytestream += len(residual_bytestream).to_bytes(4, 'little')
    bytestream += bytes(residual_bytestream)
    f.write(bytestream)

    bytelen = len(bytestream)
    print(f'\tBytestream: {naturalsize(bytelen)}.')

    print(f'\tTotal len: {naturalsize(metalen+codelen+bytelen)}.\n')

    f.close()

    with open('args.out', 'wb') as f:
        pickle.dump(args, f)


def pred_huff_run_dec(comp_file):
    '''
    Predictive Huffman with Golomb Run-Length Decoder

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
            metadata for compression; no extra compression metadata is needed
            for the predictive Huffman strategy, so this is the same as
            pre_metadata
        original_shape: tuple
            shape of original data
    '''

    f = open(comp_file, 'rb')

    # Read in metastream and reconstruct some objects
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

    # Read in metadata from codestream to begin to reconstruct codebooks
    error_shape = read_shape(f)
    residual_shape = read_shape(f)
    symbol_len = readint(f, 1)
    error_code_len = readint(f, 1)
    residual_code_len = readint(f, 1)
    error_codestream_len = readint(f, 4)
    residual_codestream_len = readint(f, 4)

    # Reconstruct the Huffman codebook for the error string
    # error_code is the final codebook
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

    # Reconstruct the Huffman codebook for the residuals
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

    # Read in metadata from bytestream to begin to decode
    error_sym_padding_bits = readint(f, 1)
    error_rl_padding_bits = readint(f, 1)
    residual_padding_bits = readint(f, 1)
    error_k = readint(f, 1)
    error_sym_bytestream_len = readint(f, 4)
    error_sym_bitstream_len = error_sym_bytestream_len*8 - \
        error_sym_padding_bits
    error_sym_bytestream = f.read(error_sym_bytestream_len)
    error_rl_bytestream_len = readint(f, 4)
    error_rl_bitstream_len = error_rl_bytestream_len*8 - \
        error_rl_padding_bits
    error_rl_bytestream = f.read(error_rl_bytestream_len)
    residual_bytestream_len = readint(f, 4)
    residual_bitstream_len = residual_bytestream_len*8 - \
        residual_padding_bits
    residual_bytestream = f.read(residual_bytestream_len)

    # Reverse codebook for decoding
    error_decodings = {v: k for k, v in dict(error_code).items()}
    residual_decodings = {v: k for k, v in dict(residual_code).items()}

    # Decode error_bytestream and residual_bytestream
    error_syms = huffman_decode(error_sym_bytestream,
        error_sym_bitstream_len, error_decodings)
    error_rls = golomb_decode(error_rl_bytestream, error_rl_bitstream_len,
        error_k)
    error_stream = from_run_length(error_syms, error_rls)
    residual_stream = huffman_decode(residual_bytestream,
        residual_bitstream_len, residual_decodings)
    error_string = np.array(error_stream, dtype=dtype).reshape(error_shape)
    residuals = np.array(residual_stream, dtype=dtype).reshape(residual_shape)

    return (error_string, residuals, clf), (n_prev, pcs, ccs), \
        (n_prev, pcs, ccs), original_shape
