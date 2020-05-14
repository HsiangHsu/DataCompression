from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer

from delta_util import *

N_SAMPLES = 5
N_NEIGHBORS = 4
CROP = 4
set_logging(True)

start = timer()

# import CIFAR: 10000 32x32 RGB, 1024R, 1024G, 1024B
datapath = '../datasets/cifar-10/data_batch_1'
raw_data = unpickle(datapath)[b'data']

# format data appropriately
N = raw_data.shape[0]
W = 32
rgb_data = np.array([np.reshape(im, (3,W,W)) for im in raw_data])
random_idx = np.random.choice(N, N_SAMPLES, replace=False)
selected_data = rgb_data[random_idx]
cropped_data = np.array([[[crop_square(selected_data[i][j], W // CROP)[k]
    for i in range(len(selected_data))] for k in range(CROP**2)] for j in
    range (3)])
data = cropped_data.reshape(3, CROP**2, N_SAMPLES, (W // CROP)**2)

end = timer()
log_print('Imported and formatted data in ' +
    f'{timedelta(seconds=end-start)}.\n')

inv_orders = [[], [], []]

# LOOP THRICE
for j in range(data.shape[0]):
    print(f'COLOR {j}')
    for i in range(data.shape[1]):
        print(f'PATCH {i}')
        start = timer()

        # find nearest neighbors
        neigh = NearestNeighbors(n_neighbors=N_NEIGHBORS, radius=1.0,
            metric='minkowski', p=2)
        neigh.fit(data[j][i])

        end = timer()
        log_print('Calculated nearest neighbors in ' +
            f'{timedelta(seconds=end-start)}.')
        start = timer()

        # find minimum spanning tree
        mst = minimum_spanning_tree(
            neigh.kneighbors_graph(mode='distance')).toarray()

        end = timer()
        log_print(f'Calculated minimum spanning tree in ' +
            f'{timedelta(seconds=end-start)}.')
        start = timer()

        edges = csr_to_edges(mst)
        tree = create_tree(edges)
        order = mst_to_order(tree, edges)
        inv_orders[j].append(np.arange(len(order))[np.argsort(order)])
        ordered_data = data[j][i][order]

        end = timer()
        log_print('Ordered data in ' +
            f'{timedelta(seconds=end-start)}.\n')

        pickle.dump(ordered_data.astype(np.uint8),
            open(f'cifar_data/color_{j}/patch_{i}.p', 'wb'))

pickle.dump(np.array(inv_orders).astype(np.uint16),
    open(f'cifar_data/inv_orders.p', 'wb'))
