from matplotlib import pyplot as plt
import numpy as np

dictionary = np.load('dictionary.npy').reshape((-1, 28, 28))

for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(dictionary[i])
plt.show()
