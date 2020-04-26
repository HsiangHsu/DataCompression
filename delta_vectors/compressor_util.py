from heapq import heappush, heappop, heapify

# http://rosettacode.org/wiki/Huffman_coding#Python
#
# Takes in a dict whose keys are individual field choices and whose values are
# the counts of those choices:
# {'a': 56, 'b': 42, 'c': 10}
#
# Returns a list of [value, code] lists like the following:
# [['a', '1'], ['b', '00'], ['c', '01']]
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
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def get_freqs(values):
    frequencies = dict()
    for val in values:
        try:
            frequencies[val] += 1
        except KeyError:
            frequencies[val] = 1
    return frequencies

def coo_to_stream(coo, d_code, r_code, c_code):
    stream = ''
    for d in coo.data:
        stream += d_code[d]
    for r in coo.row:
        stream += r_code[r]
    for c in coo.col:
        stream += c_code[c]
    return stream
