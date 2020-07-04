from matplotlib import pyplot as plt
import numpy as np

dictionary = np.load('dictionary.npy').reshape((-1, 28, 28))
data = np.load('udata_in.npy').reshape((-1, 28, 28))

for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(dictionary[i])
    plt.axis('off')
plt.savefig('dict.png')

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(data[100*i])
    plt.axis('off')
plt.savefig('data.png')
