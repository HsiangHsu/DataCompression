'''
predictive.py

This module contains helper functions for implementing the predictive coding
compressor.
'''

import numpy as np
from sklearn import linear_model

from datetime import timedelta
from timeit import default_timer as timer

import pickle

from matplotlib import pyplot as plt

def predictive_comp(data, element_axis, predictor, training_context,
    true_pixels, n_prev, pcs, ccs):
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    assert training_context is not None
    assert true_pixels is not None
    estimated_pixels = clf.predict(training_context)
    estimated_pixels = np.clip(estimated_pixels, 0, 255).astype(np.uint8)
    error_string = true_pixels - estimated_pixels

    residuals = np.array([], dtype=data.dtype)
    residuals = np.append(residuals, data[:n_prev].flatten())

    if ccs == 'DAB':
        for img in data[2:]:
            residuals = np.append(residuals, img[0,:])
            residuals = np.append(residuals, img[1:,0])
    else:
        print(f'Current context string {ccs} unsupported by compressor.')
        exit()

    assert residuals.shape[0] + error_string.shape[0] == np.prod(data.shape)

    return (error_string, residuals, clf), None, data.shape

def predictive_decomp(error_string, residuals, predictor, n_prev, pcs, ccs,
    original_shape):

    dtype = error_string.dtype
    minval = np.iinfo(dtype).min
    maxval = np.iinfo(dtype).max

    data = np.empty(original_shape, dtype=dtype)
    errors = reshape_errors(error_string, original_shape, n_prev, ccs)
    r_start, c_start = get_pred_range(ccs)

    for n in range(n_prev):
        to_pop = original_shape[1] * original_shape[2]
        data[n], residuals = residuals[:to_pop].reshape(original_shape[1],
            original_shape[2]), residuals[to_pop:]

    for n in range(n_prev, original_shape[0]):
        data, residuals = load_residuals(data, residuals, original_shape, n,
            ccs)

        # Run predictor over remaining pixels
        for r in range(r_start, original_shape[1]):
            for c in range(c_start, original_shape[2]):
                context = get_context(data, n_prev, pcs, ccs, n, r, c)
                prediction = predictor.predict(context)
                prediction = np.clip(prediction, minval, maxval).astype(dtype)
                data[n,r,c] = prediction + errors[n-n_prev, r-r_start,
                    c-c_start]

    assert len(residuals) == 0, \
        f'Not all residuals were consumed: {len(residuals)} pixels leftover.'

    return data

def reshape_errors(error_string, original_shape, n_prev, ccs):
    if ccs == 'DAB':
        return error_string.reshape((original_shape[0]-n_prev,
            original_shape[1]-1, original_shape[2]-1))
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit()

def get_pred_range(ccs):
    if ccs == 'DAB':
        return 1, 1
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit()

def load_residuals(data, residuals, original_shape, n, ccs):
    if ccs == 'DAB':
        to_pop = original_shape[1]
        data[n,:1], residuals = residuals[:to_pop].reshape((1, -1)), \
            residuals[to_pop:]
        to_pop = original_shape[2] - 1
        data[n,1:,:1], residuals = residuals[:to_pop].reshape((-1, 1)), \
            residuals[to_pop:]
        return data, residuals
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit()

def get_context(data, n_prev, pcs, ccs, n, r, c):
    context = np.empty((1, len(ccs)+len(pcs)*n_prev))
    if ccs == 'DAB':
        context[0, 0] = data[n,r,c-1]
        context[0, 1] = data[n,r-1,c]
        context[0, 2] = data[n,r-1,c-1]
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit()

    if pcs == 'DAB':
        for p in range(n_prev):
            context[0, len(ccs)+3*p] = data[n-(p+1),r,c-1]
            context[0, len(ccs)+3*p+1] = data[n-(p+1),r-1,c]
            context[0, len(ccs)+3*p+2] = data[n-(p+1),r-1,c-1]
    else:
        print(f'Previous context string {pcs} unsupported by decompressor.')
        exit()

    return context

