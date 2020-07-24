'''
encoders/predictive_golomb.py

This module contains the Golomb encoder for use with predictive coding.
'''

from bitstring import BitArray
from copy import deepcopy
from golomb_coding import golomb_coding
from heapq import heappush, heappop, heapify
from humanize import naturalsize
from math import ceil, log2
import numpy as np
import pickle

from utilities import readint, encode_predictor, decode_predictor, \
    write_shape, read_shape


def pred_golomb_enc(compression, pre_metadata, comp_metadata, original_shape,
    args):
    '''
    Golomb encoder

    Args:

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

    # Generate Golomb coded bytestream
    error_shape = error_string.shape
    error_string = error_string.flatten()
    residual_shape = residuals.shape
    residuals = residuals.flatten()

    error_bitstream = ''
    for error in error_string:
        error_bitstream += golomb_coding(error, args.error_k)
    padded_length = 8*ceil(len(error_bitstream)/8)
    error_padding = padded_length - len(error_bitstream);
    error_bitstream = f'{error_bitstream:0<{padded_length}}'
    error_bytestream = [int(error_bitstream[8*j:8*(j+1)], 2)
        for j in range(len(error_bitstream)//8)]

    residual_bitstream = ''
    for residual in residuals:
        residual_bitstream += golomb_coding(residual, args.residual_k)
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


def delta_huffman_dec(comp_file):
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
    n_layers = readint(f, 4)
    n_elements = readint(f, 4)
    n_points = readint(f, 4)
    comp_dtype = np.dtype(chr(readint(f, 1)))
    original_shape_len = readint(f, 1)
    shape_values = []
    for i in range(original_shape_len):
        shape_values.append(readint(f, 4))
    original_shape = tuple(shape_values)

    data_size = comp_dtype.itemsize

    pre_meta_included = readint(f, 1)
    if pre_meta_included:
        pre_meta_shape_len = readint(f, 1)
        pre_meta_shape_values = []
        for i in range(pre_meta_shape_len):
            pre_meta_shape_values.append(readint(f, 4))
        pre_meta_shape = tuple(pre_meta_shape_values)
        pre_meta_dtype = np.dtype(chr(readint(f, 1)))
        pre_meta_size = pre_meta_dtype.itemsize
        to_read = np.prod(pre_meta_shape) * pre_meta_dtype.itemsize
        pre_metadata = np.frombuffer(f.read(to_read), dtype=pre_meta_dtype)
        pre_metadata = pre_metadata.reshape(pre_meta_shape)
    else:
        pre_metadata = None

    comp_meta_included = readint(f, 1)
    if comp_meta_included:
        comp_meta_dtype = np.dtype(chr(readint(f, 1)))
        comp_meta_size = comp_meta_dtype.itemsize
        comp_metadata = np.empty((n_layers, n_elements), dtype=comp_meta_dtype)
        for i in range(n_layers):
            for j in range(n_elements):
                metadata[i][j] = readint(f, comp_meta_size)
    else:
        comp_metadata = None

    compression = np.empty((n_layers, n_elements, n_points), dtype=comp_dtype)

    for i in range(n_layers):
        codestream_len = readint(f, 4)
        symbol_len = readint(f, 1)
        code_len = readint(f, 1)

        # Reconstruct Huffman code
        codebytes_read = 0
        raw_code = []
        while codebytes_read < codestream_len:
            raw_code.append([readint(f, symbol_len), readint(f, code_len)])
            codebytes_read += symbol_len + code_len
        reconstructed_code = deepcopy(raw_code)
        codelen = raw_code[0][1]
        code = f'{0:0{codelen}b}'
        reconstructed_code[0][1] = code
        for k in range(1, len(raw_code)):
            codelen = raw_code[k][1]
            code = int(code, 2) + 1
            code = code << (codelen - raw_code[k-1][1])
            code = f'{code:0{codelen}b}'
            reconstructed_code[k][1] = code

        bytestream_len = readint(f, 4)
        padding_bits = readint(f, 1)
        bitstream_len = bytestream_len*8 - padding_bits
        bytestream = f.read(bytestream_len)

        for j in range(n_points):
            compression[i][0][j] = readint(f, data_size)

        # Corner case where there is no difference across layer
        if bytestream_len == 0:
            for j in range(1, n_elements):
                compression[i][j] = compression[i][0]
            continue

        decodings = {v: k for k, v in dict(reconstructed_code).items()}
        decoded_stream = huffman_decode(bytestream, bitstream_len, decodings)
        deltas = np.array(decoded_stream, dtype=comp_dtype)
        deltas = deltas.reshape(n_elements-1, n_points)

        for j in range(1, n_elements):
            compression[i][j] = compression[i][j-1] + deltas[j-1]

    return compression, pre_metadata, comp_metadata, original_shape


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
