import numpy as np
from numpy.random import default_rng as rng

N_COLS = 1
N_ROWS = 10

np.set_printoptions(precision=2, suppress=True)

# Generate alphas and alphabet size for each column
# alphabet_size = np.zeros((N_COLS), dtype=np.uint)
alphabet_size = 3
alpha = np.full(alphabet_size, 1)

data = np.empty((N_ROWS, N_COLS), dtype=np.uint)

for col in range(N_COLS):
    prior = rng().dirichlet(alpha)
    draws = rng().multinomial(1, prior, size=(N_ROWS))
    data_col = np.where(draws==1)[1]
    data[:, col] = data_col

with open('synthetic_data.np', 'wb') as f:
    np.save(f, data)
