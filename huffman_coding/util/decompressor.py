from bitstring import BitArray
from math import ceil


# Takes in an raw binary string (in the form of a Python binary string, which is
# a bytestring), the actual length of the raw binary (excluding padding and in
# bits), and a dict of decodings, as returned by generate_decodings.
#
# Returns a list of decoded values:
# ['a', 'b', 'c', 'b', 'c', 'a', 'a', 'b']
def decode_raw_binary(raw_binary, length, decodings):
    raw_binary_bits = BitArray(raw_binary).bin
    decoded_text = list()
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
#
# Also returns a dict whose keys are fieldnames and whose values are lists of
# the decompressed sequence of individual field choices that are present in the
# compressed dataset:
# {'field1': ['a', 'b', 'c', 'b', 'c', 'a', 'a', 'b'],
#  'field2': ['d', 'e', 'd', 'e', 'e', 'd', 'e', 'e']}
def generate_decompressions(infile):
    decodings = dict()
    categories = list()
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


# Takes in a relative filepath to a compressed dataset and a relative filepath
# to write the decompressed file to.
#
# Calls all the necessary helper functions to decompress the dataset.
def decompress_file(infile, outfile):
    categories, decompressions = generate_decompressions(infile)
    write_decompressed_file(outfile, categories, decompressions)
