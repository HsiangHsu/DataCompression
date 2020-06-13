import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng as rng
from scipy.optimize import curve_fit

N_BALLS = 10 ** 2
N_BINS = 10 ** 7

# Generate data
prior = np.full(N_BINS, 1/N_BINS)
result = rng().multinomial(N_BALLS, prior)
result = np.insert(result, 0, 1)
if not result[-1]:
    result = np.append(result, 1)
nonzero = result.nonzero()[0]
spacings = np.diff(nonzero) - 1

# Fit to exponential distribution
def expon_pdf(x, l):
    return l * np.exp(-l * x)
hist, bin_edges = np.histogram(spacings, bins=100, density=True)
edges = bin_edges[1:]
popt, _ = curve_fit(expon_pdf, edges, hist, p0=N_BALLS/N_BINS)

# Plot PDFs
fig, ax = plt.subplots()
ax.plot(edges, expon_pdf(edges, *popt),
    label='Empirical: ' + r'$\lambda$' + f' = {popt[0]:.2E}')
ax.plot(edges, expon_pdf(edges, N_BALLS/N_BINS),
    label='Theoretical: ' + r'$\lambda$' + f' = {N_BALLS/N_BINS:.2E}')
ax.set_xlabel('Spacing between Bins with Balls')
ax.set_ylabel('Probability Density')
ax.set_title(
    f'Exponential PDFs for {N_BINS:.0E} Bins and {N_BALLS:.0E} Balls', pad=20)
ax.ticklabel_format(style='sci', scilimits=(0,0))
plt.legend()
plt.show()
