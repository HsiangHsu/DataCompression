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

from utilities import readint


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
    b_clf = pickle.dumps(clf)

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
    metastream += len(b_clf).to_bytes(4, 'little')
    metastream += b_clf

    f = open('comp.out', 'wb')
    metalen = len(metastream)
    print(f'\tMetastream: {metalen} bytes')
    f.write(metastream)

    # Generate Huffman code for error string
    freqs = get_freqs(error_string)
    print(freqs)
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
        f'{n_residual_symbols} symbols')
    f.write(len(error_codestream).to_bytes(4, 'little'))
    f.write(len(residual_codestream).to_bytes(4, 'little'))
    f.write(symbol_len.to_bytes(1, 'little'))
    f.write(error_code_len.to_bytes(1, 'little'))
    f.write(residual_code_len.to_bytes(1, 'little'))
    f.write(error_codestream)
    f.write(residual_codestream)

    # Generate Huffman coded bitstream for data on this layer
    bitstream = ''
    for error in error_string:
        bitstream += error_code[error]
    for residual in residuals:
        bitstream += residual_code[residual]
    padded_length = 8*ceil(len(bitstream)/8)
    padding = padded_length - len(bitstream);
    bitstream = f'{bitstream:0<{padded_length}}'
    bytestream = [int(bitstream[8*j:8*(j+1)], 2)
        for j in range(len(bitstream)//8)]

    # The first element is not Huffman coded because it is not a delta
    bytestreamlen = len(bytestream)+5
    print(f'\tBytestream: {bytestreamlen} bytes')
    f.write(len(bytestream).to_bytes(4, 'little'))
    f.write(padding.to_bytes(1, 'little'))
    f.write(bytes(bytestream))

    print(f'\tTotal len: {metalen+codelen+bytestreamlen}\n')

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
