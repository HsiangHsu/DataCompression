import pickle
import numpy as np
from matplotlib import pyplot as plt

def unpickle(f):
    with open(f, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

data = unpickle('data_batch_1')

for N in range(10,20):
    image = np.transpose(np.reshape(data[b'data'][N], (3,32,32)), (1,2,0))
    plt.imshow(image)
    plt.title(f"{data[b'filenames'][N]}")
    plt.show()
