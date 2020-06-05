'''
encoders/delta_huffman.py

This module contains the Delta-Huffman encoder
'''

from bitstring import BitArray
from heapq import heappush, heappop, heapify
from math import ceil
import numpy as np
import pickle

from utilities import readint


def delta_huffman_enc(compression, metadata, original_shape, args):
    '''
    Delta Vector and Huffman encoder

    The differences between adjacent elements are written to disk using
    a Huffman code.

    Intended to be used after a smart ordering compressor, such as knn-mst,
    but this is not required.

    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_elements, n_points)
        metadata: numpy array
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

    n_layers = compression.shape[0]
    n_elements = compression.shape[1]
    n_points = compression.shape[2]

    # Bytestreams to be built and written
    metastream = b''

    # Metadata needed to reconstruct arrays: shape and dtype.
    # 4 bytes are used to be fit a reasonably wide range of values
    metastream += n_layers.to_bytes(4, 'little')
    metastream += n_elements.to_bytes(4, 'little')
    metastream += n_points.to_bytes(4, 'little')
    metastream += ord(compression.dtype.char).to_bytes(1, 'little')
    metastream += ord(metadata.dtype.char).to_bytes(1, 'little')
    metastream += len(original_shape).to_bytes(1, 'little')
    for i in range(len(original_shape)):
        metastream += original_shape[i].to_bytes(4, 'little')
    metastream += metadata.tobytes()

    f = open('comp.out', 'wb')
    f.write(metastream)

    for i in range(n_layers):
        bitstream = ''
        deltas = np.array([compression[i][j] - compression[i][j-1]
            for j in range(1, n_elements)])

        # Generate Huffman code for deltas on this layer
        freqs = get_freqs(deltas.flatten())
        code = huffman_encode(freqs)
        codestream = pickle.dumps(code)
        f.write(len(codestream).to_bytes(4, 'little'))
        f.write(codestream)

        # Generate Huffman coded bitstream for data on this layer
        for delta in deltas:
            for point in delta:
                bitstream += code[point]
        padded_length = 8*ceil(len(bitstream)/8)
        padding = padded_length - len(bitstream);
        bitstream = f'{bitstream:0<{padded_length}}'
        bytestream = [int(bitstream[8*j:8*(j+1)], 2)
            for j in range(len(bitstream)//8)]

        # The first element is not Huffman coded because it is not a delta
        f.write(len(bytestream).to_bytes(4, 'little'))
        f.write(padding.to_bytes(1, 'little'))
        f.write(bytes(bytestream))
        f.write(compression[i][0].tobytes())

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
        metadata: numpy array
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
    meta_dtype = np.dtype(chr(readint(f, 1)))
    original_shape_len = readint(f, 1)
    shape_values = []
    for i in range(original_shape_len):
        shape_values.append(readint(f, 4))
    original_shape = tuple(shape_values)

    data_size = comp_dtype.itemsize
    meta_size = meta_dtype.itemsize

    compression = np.empty((n_layers, n_elements, n_points), dtype=comp_dtype)
    metadata = np.empty((n_layers, n_elements), dtype=meta_dtype)

    for i in range(n_layers):
        for j in range(n_elements):
            metadata[i][j] = readint(f, meta_size)

    for i in range(n_layers):
        codestream_len = readint(f, 4)
        code = pickle.loads(f.read(codestream_len))
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

        decodings = {v: k for k, v in code.items()}
        decoded_stream = huffman_decode(bytestream, bitstream_len, decodings)
        deltas = np.array(decoded_stream, dtype=comp_dtype)
        deltas = deltas.reshape(n_elements-1, n_points)

        for j in range(1, n_elements):
            compression[i][j] = compression[i][j-1] + deltas[j-1]

    return compression, metadata, original_shape


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
    return dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))


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
