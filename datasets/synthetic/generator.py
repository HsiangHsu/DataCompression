import argparse
import numpy as np
from numpy.random import default_rng as rng


np.set_printoptions(precision=2, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=int, required=True)
parser.add_argument('-a', type=int, required=True)
args = parser.parse_args()

N_COLS = 1
N_ROWS = args.r
ALPH_SIZE = args.a

# Generate alphas and alphabet size for each column
# ALPH_SIZE = np.zeros((N_COLS), dtype=np.uint)
alpha = np.full(ALPH_SIZE, 1)

data = np.empty((N_ROWS, N_COLS), dtype=np.uint)

for col in range(N_COLS):
    prior = rng().dirichlet(alpha)
    draws = rng().multinomial(1, prior, size=(N_ROWS))
    data_col = np.where(draws==1)[1]
    data[:, col] = data_col

with open('synthetic_data.np', 'wb') as f:
    np.save(f, data)
