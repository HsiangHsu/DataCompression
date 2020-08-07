'''
utilities.py

This module includes various helper functions used by various modules
'''

from bitstring import BitArray
from copy import deepcopy
from golomb_coding import golomb_coding
from heapq import heappush, heappop, heapify
from itertools import chain, groupby, tee
from math import ceil, log2
from minimal_binary_coding import minimal_binary_coding
import numpy as np
import re
from sklearn import linear_model


def valid_pixels_from_context_strategy(img_shape, relative_indices,
    no_input_check=False):
    '''
    Returns the minimum and maximum values for rows/columns to iterate over
    within an image of dimension |img_shape| that can be sequentially decoded
    in a linear scan pattern given sufficient initial context.

    If |no_input_check|, we assume usage involves scan patterns or context that
    is "ahead of" (either to the right or below) a current pixel, which is
    possible for previous images that have been fully decoded.

    Otherwise we require relative context indices to be negative-valued in the
    row or zero in the current row but negative in the column.
    '''

    err_msg = "Impossible to satisfy passing initial context with these " + \
        "relative indices %r"
    assert no_input_check or np.all([index[0] < 0 or \
        (index[0] == 0 and index[1] < 0) for index in relative_indices]), \
        err_msg % relative_indices
    min_x = abs(min([index[0] for index in relative_indices]))
    max_x = img_shape[1] - \
        max(0, max([index[0] for index in relative_indices])) - 1
    min_y = abs(min([index[1] for index in relative_indices]))
    max_y = img_shape[0] - \
        max(0, max([index[1] for index in relative_indices])) - 1
    return min_x, max_x, min_y, max_y


def get_valid_pixels_for_predictions(img_shape, current_context_indices,
    prev_context_indices, return_tuples=False):
    '''
    Returns locations within an image of dimension |img_shape| that can be
    sequentially decoded in a linear scan pattern given sufficient initial
    context and previous context indices.

    Args:
        img_shape: tuple (x, y)
            dimensions of the ndarray for images in the dataset
        current_context_strategy: list of tuples (i, j)
            describes which pixels relative to a given location should be
            considered context in the current image
        prev_context_indices: list of tuples (i, j)
            describes which pixels relative to a given location should be
            considered context from prior images
        return_tuples: boolean
            whether the return value should be

    Returns:
        If |return_tuples|
            valid_pixels: list of (i, j) tuples
                indices from the range [min_x, max_x] x [min_y, max_y]
        Else
            (min_x, max_x, min_y, max_y): 4-tuple of ints
                inclusive indices for rows/columns that are valid to predict
                given sufficient context
    '''

    min_x_cur, max_x_cur, min_y_cur, max_y_cur = \
        valid_pixels_from_context_strategy(img_shape, current_context_indices)
    min_x_prev, max_x_prev, min_y_prev, max_y_prev = \
        valid_pixels_from_context_strategy(img_shape, prev_context_indices,
        no_input_check=True)
    min_x = max(min_x_cur, min_x_prev)
    max_x = min(max_x_cur, max_x_prev)
    min_y = max(min_y_cur, min_y_prev)
    max_y = min(max_y_cur, max_y_prev)
    if return_tuples:
        valid_pixels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                valid_pixels += [(x, y)]
        return valid_pixels
    return min_x, max_x, min_y, max_y


def name_to_context_pixels(name):
    '''
    Helper function to convert context name string to array of relative
    indices.
    '''

    if name == 'DAB':
        return [(0, -1), (-1, -1), (-1, 0)]
    if name == 'DABC':
        return [(0, -1), (-1, -1), (-1, 0), (-1, 1)]
    if name == 'DABX':
        return [(0, -1), (-1, -1), (-1, 0), (0, 0)]
    return None


def predictions_to_pixels(predictions, dtype):
    '''
    Casts the output of a model's |predictions| into pixel values that fit
    in |dtype|.
    '''

    minval, maxval = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(predictions, minval, maxval).astype(dtype)


def find_dtype(n):
    '''
    Finds the smallest numpy datatype given a maximum value that must be
    representable

    Args:
        n: int
            maximum value to represent

    Returns:
        dtype: numpy dtype
            smallest numpy dtype for n
    '''

    sizes = [8, 16, 32, 64]
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    valid_sizes = [sz for sz in sizes if sz >= int(ceil(log2(n)))]
    dtypes_index = sizes.index(min(valid_sizes))

    return dtypes[dtypes_index]


def readint(f, n):
    '''
    Reads in bytes from a file as an int

    Args:
        f: open file
            file to read from
        n: int
            number of bytes to read

    Returns:
        d: int
            integer representation of the n bytes read
    '''

    return int.from_bytes(f.read(n), byteorder='little')


def encode_predictor(clf):
    '''
    Helper function to extract the relevant parameters from a list of
    predictors to encode in a bytestream.

    Args:
        clf: list
            list of predictors to encode

    Returns:
        stream: bytestring
            encoded predictors
    '''

    stream = b''
    pred_name = str(clf[0]).split('(')[0]
    stream += len(pred_name).to_bytes(1, 'little')
    stream += pred_name.encode()

    for pred in clf:
        b_coef = pred.coef_.tobytes()
        stream += len(b_coef).to_bytes(4, 'little')
        stream += b_coef
        stream += write_shape(pred.coef_.shape)

        b_intercept = pred.intercept_.tobytes()
        stream += len(b_intercept).to_bytes(2, 'little')
        stream += b_intercept
        stream += write_shape(pred.intercept_.shape)

        if pred_name == 'SGDClassifier':
            b_classes = pred.classes_.tobytes()
            stream += len(b_classes).to_bytes(2, 'little')
            stream += b_classes

    return stream


def decode_predictor(f, n_pred):
    '''
    Helper function to read and decode a stream encoded by encode_predictor
    and recreate the predictors from that stream.

    Args:
        f: file
            file object to read from
        n_pred: int
            number of predictors to decode

    Returns:
        clf: list
            list of predictors recreated from decoded parameters
    '''

    predictors = {'LinearRegression':linear_model.LinearRegression,
        'SGDClassifier':linear_model.SGDClassifier}

    len_pred_name = readint(f, 1)
    pred_name = f.read(len_pred_name).decode()

    clf = [predictors[pred_name]() for i in range(n_pred)]
    for i in range(len(clf)):
        len_b_coef = readint(f, 4)
        b_coef = f.read(len_b_coef)
        coef_shape = read_shape(f)
        coef = np.frombuffer(b_coef).reshape(coef_shape)

        len_b_intercept = readint(f, 2)
        b_intercept = f.read(len_b_intercept)
        intercept_shape = read_shape(f)
        intercept = np.frombuffer(b_intercept).reshape(intercept_shape)

        if pred_name == 'SGDClassifier':
            len_b_classes = readint(f, 2)
            b_classes = f.read(len_b_classes)
            classes = np.frombuffer(b_classes, dtype=np.uint8)

        clf[i] = predictors[pred_name]()
        clf[i].coef_ = coef
        clf[i].intercept_ = intercept
        if pred_name == 'SGDClassifier':
            clf[i].classes_ = classes

    return clf


def write_shape(shape):
    '''
    Helper function to write a numpy array shape to a bytestream.

    Args:
        shape: tuple
            shape to write

    Returns:
        stream: bytestring
            encoded shape
    '''

    stream = b''
    stream += len(shape).to_bytes(1, 'little')
    for i in range(len(shape)):
        stream += shape[i].to_bytes(4, 'little')
    return stream


def read_shape(f):
    '''
    Helper function to read a numpy array shape encoded by write_shape.

    Args:
        f: file
            file to read from

    Returns:
        shape: tuple
            decoded shape
    '''

    ndim = readint(f, 1)
    shape_values = []
    for i in range(ndim):
        shape_values.append(readint(f, 4))
    return tuple(shape_values)


def get_freqs(values):
    '''
    Helper function to create a dictionary of frequencies of items in a list.

    Args:
        values: list
            list of values over which to compute

    Returns:
        freqs: dict
            dictionary whose keys are the symbols in values and whose values
            are the number of times the key occurs in values
    '''

    freqs = dict()
    for val in values:
        try:
            freqs[val] += 1
        except KeyError:
            freqs[val] = 1
    return freqs


def huffman_encode(symb2freq):
    '''
    Helper function to Huffman encode a dictionary of symbols and frequencies,
    as generated by get_freqs.

    Args:
        symb2freq: dict
            dictionary of symbols to frequencies

    Returns:
        canonical: list
            canonical Huffman code in the form of a list of lists, with the
            items of the form [symbol, code]
    '''

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
    '''
    Helper function to decode a Huffman encoded bytestream

    Args:
        bytestream: bytestring
            encoded stream
        length:
            unpadded length of stream
        decodings:
            Huffman codebook to use when decoding

    Returns:
        decoded_stream: list
            stream as a list of decoded symbols
    '''

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


def golomb_encode(stream, k):
    '''
    Helper function to encode a stream using a Golomb code

    Args:
        stream: list
            list of values to encode
        k: int
            Golomb parameter

    Returns:
        encoded_stream: string
            string of 0's and 1's representing a bitstring
    '''

    encoded_stream = ['0'*(val//k)+'1'+minimal_binary_coding(val%k,k)
        for val in stream]

    return ''.join(encoded_stream)


def golomb_decode(bytestream, bitstream_len, k):
    '''
    Helper function to decode a bytestream encoded using a Golomb code

    Args:
        bytestream: bytestring
            stream to decode
        bitstream_len: int
            length of unpadded stream
        k: int
            Golomb parameter

    Returns:
        decoded_stream: list
            stream as a list of decoded symbols
    '''

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


def to_run_length(stream):
    '''
    Helper function to convert a stream to a run-length encodeable form, as
    a list of symbols and a list of run lengths

    Args:
        stream: list
            list of symbols

    Returns:
        symbols: list
            list of symbols in order of occurrence
        run_lengths: list
            list of integers indicating the run-length of each symbol at the
            corresponding index in symbols
    '''

    symbols, run_lengths = zip(*[(a, len([*b])) for a, b in groupby(stream)])
    return symbols, run_lengths


def from_run_length(symbols, run_lengths):
    '''
    Helper function to go from a run-length representation to a single stream

    Args:
        symbols: list
            list of symbols in order of occurrence
        run_lengths: list
            list of integers indicating the run-length of each symbol at the
            corresponding index in symbols

    Returns:
        stream: list
            list of symbols
    '''

    return list(chain.from_iterable(
        [[a for i in range(b)] for a, b in list(zip(symbols, run_lengths))]))
