from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer

from util import *

N_SAMPLES = 10000
N_NEIGHBORS = 100
BINS = 100
CROP = 4
PLOT = False
set_logging(False)

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
log_print(f'Imported and formatted data in {timedelta(seconds=end-start)}.\n')

# LOOP THRICE
for j in range(data.shape[0]):
    print(f'COLOR {j}')
    for i in range(data.shape[1]):
        print(f'PATCH {i}')

        start = timer()

        # find nearest neighbors
        neigh = NearestNeighbors(n_neighbors=N_NEIGHBORS,
            # metric='minkowski', p=1)
            metric='hamming')
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
        rand_order = np.random.permutation(order)

        distances = order_distances(order, data[j][i])
        rand_distances = order_distances(rand_order, data[j][i])
        n_dist = len(distances)
        n_rand_dist = len(rand_distances)

        end = timer()
        log_print('Calculated orders in ' +
            f'{timedelta(seconds=end-start)}.\n')

        fig, axs = plt.subplots(2, sharex=True)

        counts, _, _ = axs[0].hist(distances, bins=BINS)
        axs[0].set_title('MST Distances')
        probs = counts/n_dist
        e = entropy(probs, base=2)

        rand_counts, _, _ = axs[1].hist(rand_distances, bins=BINS)
        axs[1].set_title('Random Distances')
        rand_probs = rand_counts/n_rand_dist
        rand_e = entropy(rand_probs, base=2)

        print(f'MST Entropy: {e}')
        print(f'Random Entropy: {rand_e}')
        print()

        if PLOT:
            plt.show()
