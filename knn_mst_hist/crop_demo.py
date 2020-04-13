import idx2numpy
import matplotlib.pyplot as plt
import numpy as np

from util import *

N_SAMPLES = 1
CROP = 4

# import MNIST
datapath = '../datasets/mnist/train-images-idx3-ubyte'
raw_data = idx2numpy.convert_from_file(datapath)

# format data appropriately
N = raw_data.shape[0]
W = raw_data.shape[1]
random_idx = np.random.choice(N, N_SAMPLES, replace=False)
selected_data = raw_data[random_idx][0]

M = 100

m = np.array([[M for i in range(W//CROP)] for i in range(W//CROP)])
u = np.array([[0 for i in range(W//CROP)] for i in range(W//CROP)])
a = np.concatenate([m, u, m, u], axis=1)
b = np.concatenate([u, m, u, m], axis=1)
ch_a = np.concatenate([a, b, a, b])
ch_b = np.concatenate([b, a, b, a])
ch_c = selected_data

image = np.array([ch_b, ch_c, ch_a]).transpose(1,2,0)
plt.imshow(image)
plt.show()
