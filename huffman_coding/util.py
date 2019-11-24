from bitstring import BitArray
from csv import DictReader
from heapq import heappush, heappop, heapify
from math import ceil

# Fieldnames in the dataset
fieldnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
              'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'capital-gain', 'capital-loss', 'hours-per-week',
              'native-country', 'label']

# Fieldnames that should be excluded from the compressed file
continuous_fields = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                     'capital-loss', 'hours-per-week']


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


# Takes in a csv.DictReader object which has been initialized with the input
# file and the fieldnames (see compress_file()).
#
# Returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are individual field choices and whose values are the counts of
# those choices:
# {'field1': {'a': 56, 'b': 42, 'c': 10}, 'field2': {'d' 27, 'e': 40}}
#
# Note that this function also adds an 'endline' field choice for every field,
# in order to mark the end of a field in the compressed data file.
def generate_frequencies(csv_reader):
    frequencies = dict()
    for row in csv_reader:
        for category, value in row.items():
            if category in continuous_fields:
                continue
            try:
                frequencies[category][value] += 1
            except KeyError:
                if category in frequencies:
                    frequencies[category][value] = 1
                else:
                    frequencies[category] = dict()
                    try:
                        frequencies[category][value] = 1
                    except TypeError:
                        frequencies[category][value[0]] = 1
    for category in frequencies.keys():
        frequencies[category]['endline'] = 0

    return frequencies


# Takes in a dict of frequencies, as returned by generate_frequencies().
#
# Returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are individual field choices and whose values are their Huffman
# codings, as returned by huffman_encode():
# {'field1': {'a': '1', 'b': '00', 'c': '01'}, 'field2': {'d': '0', 'e': '1'}
def generate_encodings(frequencies):
    encodings = dict()
    for category, values in frequencies.items():
        encodings[category] = dict(huffman_encode(values))
    return encodings


# Takes in a dict of encodings, as returned by generate_encodings().
#
# Returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are Huffman codings and whose values are the corresponding field
# choices:
# {'field1': {'1': 'a', '00': 'b', '01': 'c'}, 'field2': {'0': 'd', '1': 'e'}
def generate_decodings(encodings):
    decodings = dict()
    for category, values in encodings.items():
        decodings[category] = {v: k for k, v in values.items()}
    return decodings


# Takes in a properly initialzed csv.DictReader object (see compress_file())
# and a dict of encodings, as returned by generate_encodings().
#
# Returns a dict whose keys are fieldnames and whose values are raw strings of
# 0's and 1's that represent that Huffman coding for the entire sequence of all
# rows in the dataset for that fieldname:
# {'field1': '1000100011100', 'field2': '01011011'}
def generate_raw_binaries(csv_reader, encodings):
    raw_binaries = {category: '' for category in encodings}
    for row in csv_reader:
        for category, value in row.items():
            if category in continuous_fields:
                continue
            raw_binaries[category] += encodings[category][value]
    for category in raw_binaries.keys():
        raw_binaries[category] += encodings[category]['endline']
    return raw_binaries


# Takes in an raw binary string (in the form of a Python binary string, which is
# a bytestring), the actual length of the raw binary (excluding padding and in
# bits), and a dict of decodings, as returned by generate_decodings.
#
# Returns a list of decoded values:
# ['a', 'b', 'c', 'b', 'c', 'a', 'a', 'b']
def decode_raw_binary(raw_binary, length, decodings):
    raw_binary_bits = BitArray(raw_binary).bin
    decoded_text = []
    index = 0
    while (index < length):
        possible_encoding = raw_binary_bits[index]
        possible_end_index = index + 1
        while (possible_encoding not in decodings):
            possible_end_index += 1
            possible_encoding = raw_binary_bits[index:possible_end_index]
        decoded_text.append(decodings[possible_encoding])
        index = possible_end_index
    return decoded_text


# Takes in a relative filepath to a compressed dataset.
#
# Returns a list of the fieldnames in the compressed dataset:
# ['field1', 'field2']
# and a dict whose keys are fieldnames and whose values are lists of the
# decompressed sequence of individual field choices that are present in the
# compressed dataset:
# {'field1': ['a', 'b', 'c', 'b', 'c', 'a', 'a', 'b'],
#  'field2': ['d', 'e', 'd', 'e', 'e', 'd', 'e', 'e']}
def generate_decompressions(infile):
    decodings = dict()
    categories = []
    decompressions = dict()

    with open(infile, 'rb') as f:
        for rawline in f:
            line = rawline.decode('utf-8')
            if line == '***\n':
                break

            category, values = line.split(':')
            decodings[category] = dict()
            categories.append(category)
            for pair in values.split():
                value, code = pair.split(',')
                if value == 'len':
                    decodings[category][value] = code
                else:
                    decodings[category][code] = value

        for category in categories:
            length = int(decodings[category]['len'])
            bytes_to_read = ceil(length / 8)
            line = f.read(bytes_to_read)
            decompressions[category] = decode_raw_binary(line, length, decodings[category])

    return categories, decompressions


# Takes in a relative filepath to write a compressed file to, a dict of
# encodings, as returned by generate_encodings, and a dict of raw binaries, as
# returned by generate_raw_binaries.
#
# First, a uncompressed header is written which contains all the encodings used
# in the compression as well as the length of each compressed fieldname across
# the entire dataset. This metadata is necessary for decompressing the file.
#
# Then, the concatenation of all the raw binaries is written to the file. This
# is done by bytes, not bits, so if a particular fieldname has a length that is
# not a multiple of 8, then it is padded with 0's at the end.
def write_compressed_file(outfile, encodings, raw_binaries):
    with open(outfile, 'w') as f:
        for category in encodings.keys():
            f.write(f"{category}: ")
            for value, code in encodings[category].items():
                f.write(f"{value},{code} ")
            f.write(f"len,{len(raw_binaries[category])}")
            f.write("\n")
        f.write("***\n")

    with open(outfile, 'ab') as f:
        for category in raw_binaries:
            index = 0
            length = len(raw_binaries[category])
            remainder = length % 8
            while (index < (length - remainder)):
                f.write(bytes([int(raw_binaries[category][index:index+8], base=2)]))
                index += 8
            if (remainder > 0):
                f.write(bytes([int(raw_binaries[category][index:index+8], base=2) << (8 - remainder)]))


# Takes in a relative path to a file to write the compressed file to,
# a list of categories, and a list of decompressions, both as returned by
# generate_decompressions().
#
# The categories are looped through (in the same order as they are presented in
# the compressed file's header) and all the decompressions at the same index
# are pulled out of the decompressions dict. All the values with the same index
# are printed on the same line. In other words, the 0th value for field1 and
# the 0th value for field 2 are printed on the same line, separeted by a comma
# and a space. Rows are separated by newlines. The last line is blank.
#
# The decompressed file has the exact same structure as the original file.
def write_decompressed_file(outfile, categories, decompressions):
    with open(outfile, 'w') as f:
        lines = len(decompressions[categories[0]]) - 1
        index = 0
        while (index < lines):
            for category in categories:
                f.write(decompressions[category][index])
                if ((categories.index(category) + 1) == len(categories)):
                    f.write('\n')
                else:
                    f.write(', ')
            index += 1
        f.write('\n')


# Takes in a relative filepath to a raw dataset and a relative filepath to
# write the compressed file to.
#
# Calls all the necessary helper functions to compress the dataset.
def compress_file(infile, outfile):
    with open(infile) as f:
        csv_reader = DictReader(f, fieldnames=fieldnames,
                                    skipinitialspace=True)
        f.seek(0)
        frequencies = generate_frequencies(csv_reader)

        encodings = generate_encodings(frequencies)
        decodings = generate_decodings(encodings)

        f.seek(0)
        raw_binaries = generate_raw_binaries(csv_reader, encodings)

        write_compressed_file(outfile, encodings, raw_binaries)


# Takes in a relative filepath to a compressed dataset and a relative filepath
# to write the decompressed file to.
#
# Calls all the necessary helper functions to decompress the dataset.
def decompress_file(infile, outfile):
    categories, decompressions = generate_decompressions(infile)
    write_decompressed_file(outfile, categories, decompressions)
