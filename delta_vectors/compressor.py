import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

inv_orders = pickle.load(open('patches_60000/inv_orders.p', 'rb'))

for i in range(16):
    data = pickle.load(open(f'patches_60000/patch_{i}.p', 'rb'))
    data = data.astype(np.uint8)
    N = data.shape[0]
    data = data.reshape(N, 7, 7)
    deltas = np.array([data[i]-data[i-1] for i in range(1,N)])
    coos = [coo_matrix(delta) for delta in deltas]

    with open(f'comp_patches/patch_{i}', 'wb') as f:
        f.write(data[0].tobytes())
        for coo in coos:
            f.write(coo.size.to_bytes(1, 'little'))
            f.write(coo.data.tobytes())
            f.write(coo.row.astype(np.uint8).tobytes())
            f.write(coo.col.astype(np.uint8).tobytes())

    with open(f'comp_patches/inv_order_{i}', 'wb') as f:
        for pos in inv_orders[i].astype(np.uint16):
            f.write(pos.tobytes())
