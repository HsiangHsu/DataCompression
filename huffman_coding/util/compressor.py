from csv import DictReader
from heapq import heappush, heappop, heapify
from .quantizer import lloyd_max_quantizer, quantize

# Fieldnames in the dataset
fieldnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
              'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'capital-gain', 'capital-loss', 'hours-per-week',
              'native-country', 'label']

# Fieldnames that should be discretized before compression
continuous_fields = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                     'capital-loss', 'hours-per-week']

NUM_LM_ITERS = 64
NUM_LM_BINS  = 16


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
            try:
                frequencies[category][value] += 1
            except KeyError:
                if category in frequencies:
                    # First occurrence of discrete value
                    frequencies[category][value] = 1
                else:
                    # First occurrence of discrete category
                    frequencies[category] = dict()
                    frequencies[category][value] = 1
    for category in frequencies.keys():
        frequencies[category]['endline'] = 0

    return frequencies


# Takes in a dict of frequencies, as returned by generate_frequencies().
#
# Returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are individual field choices, which, if they belong to a
# continuous field, have been quantized using a Lloyd-Max quantizer, and whose
# values are the counts of those choices. This dict looks similar to the one
# returned by generate_frequencies() and includes the 'endline' dummy value.
#
# Also returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are 'A' and 'B' and whose values are the Lloyd-Max quantization
# parameters which were used to discretize the corresponding field:
# {'field1': {'A': [1, 5, 9], 'B': [3, 7]}}
def discretize_frequencies(frequencies):
    discretized_frequencies = dict()
    quantization_parameters = dict()

    print("Quantizing continuous fields:")

    for category, freqs in frequencies.items():
        if category in continuous_fields:
            # Unpack frequency dict into list of values to run Lloyd-Max
            original_values = list()
            for value in freqs:
                for i in range(freqs[value]):
                    original_values.append(int(value))

            # Run Lloyd-Max repeatedly to try to improve MSE
            saved_A = None
            saved_B = None
            lowest_mse = 2**64
            if category == 'fnlwgt':
                iter_count = 1
            else:
                iter_count = NUM_LM_ITERS
            for i in range(iter_count):
                A, B, mse = lloyd_max_quantizer(original_values, NUM_LM_BINS)
                # If quantizer returns -1 as MSE, then there was an error with
                # that run and the result should not be counted
                if mse == -1:
                    continue
                if mse < lowest_mse:
                    lowest_mse = mse
                    saved_A, saved_B = A, B
            quantization_parameters[category] = {'A': saved_A, 'B': saved_B}

            # Log information about data loss
            print(f"\t{category}, MSE: {lowest_mse}")

            # Quantize frequency dict using best Lloyd-Max results
            quantized_freqs = dict()
            for value, freq in freqs.items():
                if value != 'endline':
                    quantized = quantize(int(value), saved_A, saved_B)
                    try:
                        quantized_freqs[str(quantized)] += freq
                    except KeyError:
                        quantized_freqs[str(quantized)] = freq
            quantized_freqs['endline'] = 0
            discretized_frequencies[category] = quantized_freqs

        else:
            # If the category is not continuous, just copy it
            discretized_frequencies[category] = freqs

    return discretized_frequencies, quantization_parameters


# Takes in a dict of discretized frequencies, as returned by
# discretize_frequencies().
#
# Returns a dict whose keys are fieldnames and whose values are also dicts,
# whose keys are individual field choices and whose values are their Huffman
# codings, as returned by huffman_encode():
# {'field1': {'a': '1', 'b': '00', 'c': '01'}, 'field2': {'d': '0', 'e': '1'}
def generate_encodings(discretized_frequencies):
    encodings = dict()
    for category, values in discretized_frequencies.items():
        encodings[category] = dict(huffman_encode(values))
    return encodings


# Takes in a properly initialzed csv.DictReader object (see compress_file()),
# a dict of encodings, as returned by generate_encodings(), and a dict of
# quantization parameters, as returned by discretize_frequencies().
#
# Returns a dict whose keys are fieldnames and whose values are raw strings of
# 0's and 1's that represent that Huffman coding for the entire sequence of all
# rows in the dataset for that fieldname:
# {'field1': '1000100011100', 'field2': '01011011'}
def generate_compressions(csv_reader, encodings, quantization_parameters):
    raw_binaries = {category: '' for category in encodings}

    for row in csv_reader:
        for category, value in row.items():
            if category in continuous_fields:
                # If the category is continuous, the true value must be
                # quantized using the same Lloyd-Max quantization parameters as
                # were used to generate the encodings.
                A = quantization_parameters[category]['A']
                B = quantization_parameters[category]['B']
                raw_binaries[category] += \
                    encodings[category][str(quantize(int(value), A, B))]
            else:
                raw_binaries[category] += encodings[category][value]

    for category in raw_binaries.keys():
        raw_binaries[category] += encodings[category]['endline']

    return raw_binaries


# Takes in a relative filepath to write a compressed file to, a dict of
# encodings, as returned by generate_encodings, and a dict of raw binaries, as
# returned by generate_compressions.
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
        discretized_frequencies, quantization_parameters = \
            discretize_frequencies(frequencies)

        encodings = generate_encodings(discretized_frequencies)

        f.seek(0)
        raw_binaries = generate_compressions(csv_reader, encodings,
                                             quantization_parameters)

        write_compressed_file(outfile, encodings, raw_binaries)
