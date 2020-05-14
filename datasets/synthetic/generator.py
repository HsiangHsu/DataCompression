import numpy as np
from numpy.random import default_rng as rng
import matplotlib.pyplot as plt
import pandas as pd

P = [
    [1/3, 1/3, 1/3],
    [1/4, 1/2, 1/4],
    [1/2, 1/2]
]

covariances = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])*10

means = np.array([50, 50, 50, 50])
data = rng().multivariate_normal(means, covariances, size=100)
data = data.astype(np.uint8)

with open('synthetic_data.np', 'wb') as f:
    np.save(f, data)
