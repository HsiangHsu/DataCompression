'''
predictive.py

This module contains helper functions for implementing the predictive coding
compressor.
'''
import numpy as np
from utilities import name_to_context_pixels, predictions_to_pixels, \
    get_valid_pixels_for_predictions

def predictive_comp(data, element_axis, predictors, training_context,
    true_pixels, n_prev, pcs, ccs, mode):
    assert training_context is not None
    assert true_pixels is not None

    n_pred = true_pixels.shape[0]
    dtype = data.dtype

    if true_pixels.ndim == 3:
        # RGB triples
        assert mode == 'single'
        to_reshape = (-1, true_pixels.shape[-1])
        residuals = np.empty((0, true_pixels.shape[-1]), dtype=dtype)
    elif true_pixels.ndim == 2:
        if mode == 'single':
            # Grayscale singles
            to_reshape = (-1,)
            residuals = np.empty((0,), dtype=dtype)
        elif mode == 'triple':
            # RGB singles
            to_reshape = (-1, true_pixels.shape[0])
            residuals = np.empty((0, true_pixels.shape[0]), dtype=dtype)
    else:
        print('Bad shape for true_pixels.')
        exit(-1)
    error_string = np.empty(true_pixels.shape, dtype=dtype)

    # Build error string
    remaining_samples_to_predict = true_pixels.shape[1]
    start_index = 0
    while remaining_samples_to_predict > 0:
        predict_batch_size = min(remaining_samples_to_predict, 1000)
        s = start_index
        e = start_index + predict_batch_size
        for i in range(n_pred):
            predictions = predictors[i].predict(training_context[s:e])
            estimated_pixels = predictions_to_pixels(predictions, dtype)
            error_string[i][s:e] = true_pixels[i][s:e] - estimated_pixels
        start_index += predict_batch_size
        remaining_samples_to_predict -= predict_batch_size

    # Build residuals
    residuals = np.append(residuals,
        data[:n_prev].reshape(to_reshape), axis=0)
    current_context_indices = name_to_context_pixels(ccs)
    prev_context_indices = name_to_context_pixels(pcs)
    r_start, r_end, c_start, c_end = \
        get_valid_pixels_for_predictions(data[0].shape,
            current_context_indices, prev_context_indices)
    r_end += 1
    c_end += 1
    for img in data[n_prev:]:
        residuals = np.append(residuals,
            img[:r_start,:].reshape(to_reshape), axis=0)
        residuals = np.append(residuals,
            img[r_start:,:c_start].reshape(to_reshape), axis=0)
        residuals = np.append(residuals,
            img[r_end:,c_start:].reshape(to_reshape), axis=0)
        residuals = np.append(residuals,
            img[r_start:r_end,c_end:].reshape(to_reshape), axis=0)

    if true_pixels.ndim == 3:
        assert residuals.shape[0] + error_string.shape[1] == \
            np.prod(data.shape[:-1])
    elif true_pixels.ndim == 2:
        if mode == 'single':
            assert residuals.shape[0] + error_string.shape[1] == \
                np.prod(data.shape)
        elif mode == 'triple':
            assert residuals.shape[0] + error_string.shape[1] == \
                np.prod(data.shape[:-1])

    return (error_string, residuals, predictors), None, data.shape


def predictive_decomp(error_string, residuals, predictors, n_prev, pcs, ccs,
    original_shape, mode):

    dtype = error_string.dtype
    data = np.empty(original_shape, dtype=dtype)
    current_context_indices = name_to_context_pixels(ccs)
    prev_context_indices = name_to_context_pixels(pcs)
    r_start, r_end, c_start, c_end = \
        get_valid_pixels_for_predictions(original_shape[1:],
            current_context_indices, prev_context_indices)
    # Add 1 because these are used for iteration, not slicing
    r_end += 1
    c_end += 1

    if error_string.ndim == 3:
        # RGB triples
        to_reshape = (1, original_shape[0]-n_prev, r_end-r_start, \
            c_end-c_start, 3)
    elif error_string.ndim == 2:
        if mode == 'single':
            # Grayscale singles
            to_reshape = (1, original_shape[0]-n_prev, r_end-r_start, \
            c_end-c_start)
        elif mode == 'triple':
            # RGB singles
            to_reshape = (3, original_shape[0]-n_prev, r_end-r_start, \
            c_end-c_start)
    errors = error_string.reshape(to_reshape)

    to_pop = n_prev * original_shape[1] * original_shape[2]
    data[:n_prev], residuals = residuals[:to_pop].reshape((n_prev,
        *data.shape[1:])), residuals[to_pop:]

    for n in range(n_prev, original_shape[0]):
        data, residuals = load_residuals(data, residuals, n, r_start, r_end,
            c_start, c_end)

        # Run predictor over remaining pixels
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                for i in range(len(predictors)):
                    context = get_context(data, n_prev, pcs, ccs, n, r, c)
                    prediction = predictors[i].predict(context)
                    prediction = predictions_to_pixels(prediction, dtype)
                    if mode == 'triple':
                        data[n,r,c,i] = prediction + errors[i, n-n_prev,
                            r-r_start, c-c_start]
                    elif mode == 'single':
                        data[n,r,c] = prediction + errors[i, n-n_prev,
                            r-r_start, c-c_start]

    assert len(residuals) == 0, \
        f'Not all residuals were consumed: {len(residuals)} pixels leftover.'

    return data


def load_residuals(data, residuals, n, r_start, r_end, c_start, c_end):
    nr = r_start
    nc = data.shape[2]
    to_pop = nr*nc
    if data.ndim == 4:
        to_reshape = (nr, nc, data.shape[-1])
    else:
        to_reshape = (nr, nc)
    data[n,:r_start,:], residuals = \
        residuals[:to_pop].reshape(to_reshape), residuals[to_pop:]

    nr = data.shape[1] - r_start
    nc = c_start
    to_pop = nr*nc
    if data.ndim == 4:
        to_reshape = (nr, nc, data.shape[-1])
    else:
        to_reshape = (nr, nc)
    data[n,r_start:,:c_start], residuals = \
        residuals[:to_pop].reshape(to_reshape), residuals[to_pop:]

    nr = data.shape[1] - r_end
    nc = data.shape[2] - c_start
    to_pop = nr*nc
    if data.ndim == 4:
        to_reshape = (nr, nc, data.shape[-1])
    else:
        to_reshape = (nr, nc)
    data[n,r_end:,c_start:], residuals = \
        residuals[:to_pop].reshape(to_reshape), residuals[to_pop:]

    nr = r_end - r_start
    nc = data.shape[2] - c_end
    to_pop = nr*nc
    if data.ndim == 4:
        to_reshape = (nr, nc, data.shape[-1])
    else:
        to_reshape = (nr, nc)
    data[n,r_start:r_end,c_end:], residuals = \
        residuals[:to_pop].reshape(to_reshape), residuals[to_pop:]

    return data, residuals


def get_context(data, n_prev, pcs, ccs, n, r, c):
    if data.ndim == 4:
        context = np.empty((1, len(ccs)+len(pcs)*n_prev, data.shape[-1]))
    else:
        context = np.empty((1, len(ccs)+len(pcs)*n_prev))
    if ccs == 'DAB':
        context[0, 0] = data[n,r,c-1].flatten()
        context[0, 1] = data[n,r-1,c-1].flatten()
        context[0, 2] = data[n,r-1,c].flatten()
    elif ccs == 'DABC':
        context[0, 0] = data[n,r,c-1].flatten()
        context[0, 1] = data[n,r-1,c-1].flatten()
        context[0, 2] = data[n,r-1,c].flatten()
        context[0, 3] = data[n,r-1,c+1].flatten()
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit(-1)

    if pcs == 'DAB':
        for p in range(n_prev):
            context[0, len(ccs)+3*p] = data[n-(p+1),r,c-1].flatten()
            context[0, len(ccs)+3*p+1] = data[n-(p+1),r-1,c-1].flatten()
            context[0, len(ccs)+3*p+2] = data[n-(p+1),r-1,c].flatten()
    elif pcs == 'DABC':
        for p in range(n_prev):
            context[0, len(ccs)+4*p] = data[n-(p+1),r,c-1].flatten()
            context[0, len(ccs)+4*p+1] = data[n-(p+1),r-1,c-1].flatten()
            context[0, len(ccs)+4*p+2] = data[n-(p+1),r-1,c].flatten()
            context[0, len(ccs)+4*p+3] = data[n-(p+1),r-1,c+1].flatten()
    else:
        print(f'Previous context string {pcs} unsupported by decompressor.')
        exit(-1)

    return context.reshape((1, -1))
