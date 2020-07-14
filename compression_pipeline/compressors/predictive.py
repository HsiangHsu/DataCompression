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
    true_pixels):
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    try:
        training_context = np.load('training_context.npy')
        true_pixels = np.load('true_pixels.npy')
    except FileNotFoundError:
        assert training_context is not None
        assert true_pixels is not None
    estimated_pixels = clf.predict(training_context)
    estimated_pixels = np.clip(estimated_pixels, 0, 255).astype(np.uint8)
    error_string = true_pixels - estimated_pixels
    # plt.hist(error_string)
    # plt.show()

    # TODO: Not hardcoded 2 imgs, 1 row, 1 col

    residuals = np.array([], dtype=data.dtype)
    residuals = np.append(residuals, data[:2].flatten())
    for img in data[2:]:
        residuals = np.append(residuals, img[0,:])
        residuals = np.append(residuals, img[1:,0])
    assert(residuals.shape[0] + error_string.shape[0] == np.prod(data.shape))

    return (error_string, residuals, clf), None, data.shape
