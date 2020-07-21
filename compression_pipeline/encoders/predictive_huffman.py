'''
encoders/predictive_huffman.py

This module contains the Huffman encoder for use with predictive coding.
'''

from bitstring import BitArray
from copy import deepcopy
from heapq import heappush, heappop, heapify
from math import ceil, log2
import numpy as np
import pickle
from sklearn import linear_model

from utilities import readint, rgb_to_int, int_to_rgb


def pred_huffman_enc(compression, pre_metadata, comp_metadata, original_shape,
    args):
    '''
    Huffman encoder

    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_elements, n_points)
        metadata:
            metadata for compression (not necessarily the same metadata
            that is returned by the loader), of shape
            (n_layers, n_elements); since this encoder is intended
            to be used with smart orderings, this will probably be inverse
            orderings of some sort
        original_shape: tuple
            shape of original data
        args: dict
            all command line arguments passed to driver_compress.py

    Returns:
        None
    '''

    error_string, residuals, clf = compression
    n_errors = error_string.shape[0]
    n_residuals = residuals.shape[0]
    n_prev = pre_metadata[0]
    pcs = pre_metadata[1]
    ccs = pre_metadata[2]

    # Bytestreams to be built and written
    metastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
    metastream += n_errors.to_bytes(4, 'little')
    metastream += n_residuals.to_bytes(4, 'little')
    metastream += ord(error_string.dtype.char).to_bytes(1, 'little')
    metastream += len(original_shape).to_bytes(1, 'little')
    for i in range(len(original_shape)):
        metastream += original_shape[i].to_bytes(4, 'little')
    metastream = encode_predictor(metastream, clf)
    metastream += n_prev.to_bytes(1, 'little')
    metastream += len(pcs).to_bytes(1, 'little')
    metastream += pcs.encode()
    metastream += len(ccs).to_bytes(1, 'little')
    metastream += ccs.encode()

    f = open('comp.out', 'wb')
    metalen = len(metastream)
    print(f'\tMetastream: {metalen} bytes.')
    f.write(metastream)

    # RGB data
    if len(original_shape) == 4 and original_shape[-1] == 3:
        error_string = rgb_to_int(error_string)
        residuals = rgb_to_int(residuals)

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

    codelen = len(residual_codestream)+len(error_codestream)+11
    print(f'\tCodestream: {codelen} bytes, {n_error_symbols} / ' + \
        f'{n_residual_symbols} symbols.')
    f.write(len(error_codestream).to_bytes(4, 'little'))
    f.write(len(residual_codestream).to_bytes(4, 'little'))
    f.write(symbol_len.to_bytes(1, 'little'))
    f.write(error_code_len.to_bytes(1, 'little'))
    f.write(residual_code_len.to_bytes(1, 'little'))
    f.write(error_codestream)
    f.write(residual_codestream)

    # Generate Huffman coded bitstream for data on this layer
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

    # The first element is not Huffman coded because it is not a delta
    bytestreamlen = len(error_bytestream+residual_bytestream)+5
    print(f'\tBytestream: {bytestreamlen} bytes.')
    f.write(len(error_bytestream).to_bytes(4, 'little'))
    f.write(error_padding.to_bytes(1, 'little'))
    f.write(bytes(error_bytestream))
    f.write(len(residual_bytestream).to_bytes(4, 'little'))
    f.write(residual_padding.to_bytes(1, 'little'))
    f.write(bytes(residual_bytestream))

    print(f'\tTotal len: {metalen+codelen+bytestreamlen}.\n')

    f.close()

    with open('args.out', 'wb') as f:
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
    n_errors = readint(f, 4)
    n_residuals = readint(f, 4)
    dtype = np.dtype(chr(readint(f, 1)))
    dsize = dtype.itemsize
    original_shape_len = readint(f, 1)
    shape_values = []
    for i in range(original_shape_len):
        shape_values.append(readint(f, 4))
    original_shape = tuple(shape_values)
    clf = decode_predictor(f)
    n_prev = readint(f, 1)
    len_pcs = readint(f, 1)
    pcs = f.read(len_pcs).decode()
    len_ccs = readint(f, 1)
    ccs = f.read(len_ccs).decode()

    # Reconstruct Huffman code
    error_codestream_len = readint(f, 4)
    residual_codestream_len = readint(f, 4)
    symbol_len = readint(f, 1)
    error_code_len = readint(f, 1)
    residual_code_len = readint(f, 1)

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

    error_bytestream_len = readint(f, 4)
    error_padding_bits = readint(f, 1)
    error_bitstream_len = error_bytestream_len*8 - error_padding_bits
    error_bytestream = f.read(error_bytestream_len)

    residual_bytestream_len = readint(f, 4)
    residual_padding_bits = readint(f, 1)
    residual_bitstream_len = residual_bytestream_len*8 - residual_padding_bits
    residual_bytestream = f.read(residual_bytestream_len)

    error_string = np.empty((n_errors,), dtype=dtype)
    residuals = np.empty((n_residuals,), dtype=dtype)

    error_decodings = {v: k for k, v in dict(error_code).items()}
    residual_decodings = {v: k for k, v in dict(residual_code).items()}
    error_decoded_stream = huffman_decode(error_bytestream,
        error_bitstream_len, error_decodings)
    residual_decoded_stream = huffman_decode(residual_bytestream,
        residual_bitstream_len, residual_decodings)

    error_string = np.array(error_decoded_stream, dtype=np.uint)
    residuals = np.array(residual_decoded_stream, dtype=np.uint)

    # RGB data
    if len(original_shape) == 4 and original_shape[-1] == 3:
        error_string = int_to_rgb(error_string)
        residuals = int_to_rgb(residuals)
    else:
        error_string = error_string.astype(np.uint8)
        residuals = residuals.astype(np.uint8)

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

def encode_predictor(metastream, clf):
    pred_name = str(clf).split('(')[0]
    metastream += len(pred_name).to_bytes(1, 'little')
    metastream += pred_name.encode()

    b_coef = clf.coef_.tobytes()
    metastream += len(b_coef).to_bytes(2, 'little')
    metastream += b_coef
    metastream += len(clf.coef_.shape).to_bytes(1, 'little')
    for i in range(len(clf.coef_.shape)):
        metastream += clf.coef_.shape[i].to_bytes(2, 'little')

    b_intercept = clf.intercept_.tobytes()
    metastream += len(b_intercept).to_bytes(2, 'little')
    metastream += b_intercept
    metastream += len(clf.intercept_.shape).to_bytes(1, 'little')
    for i in range(len(clf.intercept_.shape)):
        metastream += clf.intercept_.shape[i].to_bytes(2, 'little')

    if pred_name == 'SGDClassifier':
        b_classes = clf.classes_.tobytes()
        metastream += len(b_classes).to_bytes(2, 'little')
        metastream += b_classes

    return metastream

def decode_predictor(f):
    predictors = {'LinearRegression':linear_model.LinearRegression,
        'SGDClassifier':linear_model.SGDClassifier}

    len_pred_name = readint(f, 1)
    pred_name = f.read(len_pred_name).decode()

    len_b_coef = readint(f, 2)
    b_coef = f.read(len_b_coef)
    len_coef_shape = readint(f, 1)
    shape_values = []
    for i in range(len_coef_shape):
        shape_values.append(readint(f, 2))
    coef_shape = tuple(shape_values)
    coef = np.frombuffer(b_coef).reshape(coef_shape)

    len_b_intercept = readint(f, 2)
    b_intercept = f.read(len_b_intercept)
    len_intercept_shape = readint(f, 1)
    shape_values = []
    for i in range(len_intercept_shape):
        shape_values.append(readint(f, 2))
    intercept_shape = tuple(shape_values)
    intercept = np.frombuffer(b_intercept).reshape(intercept_shape)

    if pred_name == 'SGDClassifier':
        len_b_classes = readint(f, 2)
        b_classes = f.read(len_b_classes)
        classes = np.frombuffer(b_classes, dtype=np.uint8)

    clf = predictors[pred_name]()
    clf.coef_ = coef
    clf.intercept_ = intercept
    if pred_name == 'SGDClassifier':
        clf.classes_ = classes

    return clf
