import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree

num_samples = 10

datapath = '../../datasets/mnist/train-images-quantized-0-idx3-ubyte'
labelpath = '../../datasets/mnist/train-labels-idx1-ubyte'
raw_data = idx2numpy.convert_from_file(datapath)[0:num_samples]

patch_sz = 7
patch_num = 3

def plot_figures(figures, nrows=1, ncols=1):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for figure, ax in zip(figures, axs.ravel()):
        ax.imshow(figure, cmap='gray_r')

def crop_square(image, crop_sz):
    d = image.shape[0]
    crop_d = d // crop_sz
    cropped = np.empty((crop_d**2, crop_sz, crop_sz))
    for i in range(crop_d):
        for j in range(crop_d):
            cropped[i*crop_d+j] = image[i*crop_sz:(i+1)*crop_sz,
                                        j*crop_sz:(j+1)*crop_sz]
    return cropped

zero_cropped = np.array([crop_square(raw_data[i], patch_sz)
    for i in range(len(raw_data))])

slices = [mat.flatten() for mat in zero_cropped[:,patch_num]]

N = 5
neigh = NearestNeighbors(n_neighbors=N, metric='hamming')
neigh.fit(slices)
graph = neigh.kneighbors_graph(mode='distance')
min_span_graph = minimum_spanning_tree(graph)
print(min_span_graph)

graphable = zero_cropped[:,patch_num]
plot_figures(graphable, 1, num_samples)
plt.show()
