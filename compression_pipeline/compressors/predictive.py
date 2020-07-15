'''
knn_mst.py

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
    true_pixels):
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    training_context = np.load('training_context.npy')
    true_pixels = np.load('true_pixels.npy')
    estimated_pixels = clf.predict(training_context)
    estimated_pixels = np.clip(estimated_pixels, 0, 255).astype(np.uint8)
    error_string = true_pixels - estimated_pixels

    assert (error_string + estimated_pixels == true_pixels).all()

    # TODO: Not hardcoded 2 imgs, 1 row, 1 col

    residuals = np.array([], dtype=data.dtype)
    residuals = np.append(residuals, data[:2].flatten())
    for img in data[2:]:
        residuals = np.append(residuals, img[0,:])
        residuals = np.append(residuals, img[1:,0])

    assert residuals.shape[0] + error_string.shape[0] == np.prod(data.shape)
    exit()

    return (error_string, residuals, clf), None, data.shape

def predictive_decomp(error_string, residuals, predictor, original_shape):
    # TODO: NEED SHAPE
    # ASSUME DAB
    n_prev = 2
    n_known_rows = 1
    n_known_cols = 1

    dtype = error_string.dtype
    minval = np.iinfo(dtype).min
    maxval = np.iinfo(dtype).max

    data = np.empty(original_shape, dtype=dtype)
    errors = error_string.reshape((original_shape[0] - n_prev,
        original_shape[1] - n_known_rows, original_shape[2] - n_known_cols))

    contexts = np.empty((0,9))
    actual_contexts = np.load('training_context.npy')

    for n in range(n_prev):
        to_pop = original_shape[1] * original_shape[2]
        data[n], residuals = residuals[:to_pop].reshape(original_shape[1],
            original_shape[2]), residuals[to_pop:]

    for n in range(n_prev, original_shape[0]):
        for r in range(n_known_rows):
            to_pop = original_shape[1]
            data[n,r], residuals = residuals[:to_pop], residuals[to_pop:]
        for c in range(n_known_cols):
            to_pop = original_shape[2] - n_known_rows
            data[n,n_known_rows:,c], residuals = residuals[:to_pop], \
                residuals[to_pop:]
        for r in range(n_known_rows, original_shape[1]):
            for c in range(n_known_cols, original_shape[2]):
                context = np.empty((1+n_prev, 3))
                # TODO: This has to be iterable
                for p in range(1+n_prev):
                    context[p, 0] = data[n-p,r-1,c-1] # A
                    context[p, 1] = data[n-p,r-1,c]   # B
                    context[p, 2] = data[n-p,r,c-1]   # D
                context = context[[1,2,0]]
                context = context.flatten().reshape((1, -1))
                assert (context == actual_contexts[len(contexts)]).all(), \
                    f'{context}, {actual_contexts[len(contexts)]}, {len(contexts)}'
                contexts = np.append(contexts, context, axis=0)
                prediction = predictor.predict(context)
                prediction = np.clip(prediction, minval, maxval).astype(dtype)
                data[n,r,c] = prediction + errors[n-n_prev, r-n_known_rows,
                    c-n_known_cols]

    assert len(residuals) == 0, \
        f'Not all residuals were consumed: {len(residuals)} pixels leftover.'

    return data
