import idx2numpy
import numpy as np
from matplotlib import pyplot as plt

datapath = f"../datasets/mnist/train-images-averages-idx3-ubyte"
averages = idx2numpy.convert_from_file(datapath)

raw_binary = ''
for label in range(len(averages)):
    raw_binary += f'{np.binary_repr(31, width=5)}'
    datapath = f"../datasets/mnist/train-images-quantized-{label}-idx3-ubyte"
    uncompressed = idx2numpy.convert_from_file(datapath)

    for idx in range(len(uncompressed)):
        diff = [[0 for i in range(28)] for i in range(28)]
        diff_rows = [[] for i in range(28)]
        for i in range(28):
            for j in range(28):
                if uncompressed[idx][i][j] != averages[label][i][j]:
                    diff[i][j] = 1
                    diff_rows[i].append(j)
                else:
                    diff[i][j] = 0

        diff_string = f'{np.binary_repr(31, width=5)}'
        for i in range(len(diff_rows)):
            row = diff_rows[i]
            if row:
                diff_string += f'{np.binary_repr(30, width=5)}'
                diff_string += f'{np.binary_repr(i, width=5)}'
                prev_j = -2
                for j in row:
                    if prev_j == -2:
                        diff_string += f'{np.binary_repr(j, width=5)}'
                    elif j == prev_j + 1:
                        pass
                    else:
                        diff_string += f'{np.binary_repr(prev_j+1, width=5)}'
                        diff_string += f'{np.binary_repr(j, width=5)}'
                    if j == max(row):
                        diff_string += f'{np.binary_repr(j+1, width=5)}'
                    prev_j = j

        raw_binary += diff_string

with open('out.bin', 'wb') as f:
    i = 0
    length = len(raw_binary)
    remainder = length % 8
    while (i < (length - remainder)):
        f.write(bytes([int(raw_binary[i:i+8], base=2)]))
        i += 8
    if (remainder > 0):
        f.write(bytes([int(raw_binary[i:i+8], base=2) << (8 - remainder)]))
