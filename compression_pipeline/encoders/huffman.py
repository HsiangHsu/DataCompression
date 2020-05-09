'''
huffman.py

This module contains helper functions for implementing the Huffman encoder.
'''


from heapq import heappush, heappop, heapify
from bitstring import BitArray

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
