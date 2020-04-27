import pickle
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from scipy.sparse import coo_matrix

from compressor_util import *

inv_orders = pickle.load(open('patches_60000/inv_orders.p', 'rb'))

for i in range(16):
    data = pickle.load(open(f'patches_60000/patch_{i}.p', 'rb'))
    data = data.astype(np.uint8)
    N = data.shape[0]
    data = data.reshape(N, 7, 7)
    deltas = np.array([data[i]-data[i-1] for i in range(1,N)])
    coos = [coo_matrix(delta) for delta in deltas]

    data = []
    rows = []
    cols = []
    for coo in coos:
        data.extend(coo.data)
        rows.extend(coo.row)
        cols.extend(coo.col)

    d_code = dict(huffman_encode(get_freqs(data)))
    r_code = dict(huffman_encode(get_freqs(rows)))
    c_code = dict(huffman_encode(get_freqs(cols)))

    with open(f'huffman_comp_patches/patches/patch_{i}', 'wb') as f:
        f.write(data[0].tobytes())
        bitstream = ''
        for coo in coos:
            bitstream += f'{coo.size:0>8b}'
            bitstream += coo_to_stream(coo, d_code, r_code, c_code)
        padded_length = 8*ceil(len(bitstream)/8)
        bitstream = f'{bitstream:0<{padded_length}}'
        bytestream = [int(bitstream[8*j:8*(j+1)], 2)
            for j in range(len(bitstream)//8)]
        f.write(bytes(bytestream))

    with open(f'huffman_comp_patches/codes/code_{i}', 'w') as f:
        f.write(str(d_code)+'\n')
        f.write(str(r_code)+'\n')
        f.write(str(c_code))

    ### OLD COMPRESSION ###

    # with open(f'comp_patches/patch_{i}', 'wb') as f:
    #    f.write(data[0].tobytes())
    #    for coo in coos:
    #        f.write(coo.size.to_bytes(1, 'little'))
    #        f.write(coo.data.tobytes())
    #        f.write(coo.row.astype(np.uint8).tobytes())
    #        f.write(coo.col.astype(np.uint8).tobytes())

    # with open(f'comp_patches/inv_order_{i}', 'wb') as f:
    #    for pos in inv_orders[i].astype(np.uint16):
    #         f.write(pos.tobytes())
