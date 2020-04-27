# small = csr_matrix(big)
# pickle.dump(big, open("big.p", "wb"))
# pickle.dump(small, open("small.p", "wb"))

from datetime import timedelta
import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer

from mst_util import *

N_SAMPLES = 60000
N_NEIGHBORS = 1000
CROP = 4
set_logging(True)

start = timer()

# import MNIST
datapath = '../datasets/mnist/train-images-idx3-ubyte'
raw_data = idx2numpy.convert_from_file(datapath)

# format data appropriately
N = raw_data.shape[0]
W = raw_data.shape[1]
random_idx = np.random.choice(N, N_SAMPLES, replace=False)
selected_data = raw_data[random_idx]
cropped_data = np.array([[crop_square(selected_data[i], W // CROP)[k]
    for i in range(selected_data.shape[0])] for k in range(CROP**2)])
data = cropped_data.reshape(CROP**2, N_SAMPLES, (W // CROP)**2)

end = timer()
log_print('Imported and formatted data in ' +
    f'{timedelta(seconds=end-start)}.\n')

inv_orders = []

for i in range(data.shape[0]):
    print(f'PATCH {i}')
    start = timer()

    # find nearest neighbors
    neigh = NearestNeighbors(n_neighbors=N_NEIGHBORS, radius=1.0,
        metric='minkowski', p=2)
    neigh.fit(data[i])

    end = timer()
    log_print('Calculated nearest neighbors in ' +
        f'{timedelta(seconds=end-start)}.')
    start = timer()

    # find minimum spanning tree
    mst = minimum_spanning_tree(
        neigh.kneighbors_graph(mode='distance')).toarray()

    end = timer()
    log_print('Calculated minimum spanning tree in ' +
        f'{timedelta(seconds=end-start)}.')
    start = timer()

    edges = csr_to_edges(mst)
    tree = create_tree(edges)
    order = mst_to_order(tree, edges)
    inv_orders.append(np.arange(len(order))[np.argsort(order)])
    ordered_data = data[i][order]

    end = timer()
    log_print('Ordered data in '+
        f'{timedelta(seconds=end-start)}.\n')

    pickle.dump(ordered_data, open(f'patches_60000/patch_{i}.p', 'wb'))

pickle.dump(inv_orders, open(f'inv_orders.p', 'wb'))
