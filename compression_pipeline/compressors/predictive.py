'''
predictive.py

This module contains helper functions for implementing the predictive coding
compressor.
'''
import numpy as np
from utilities import name_to_context_pixels, convert_predictions_to_pixels, \
    get_valid_pixels_for_predictions

def predictive_comp(data, element_axis, predictor, training_context,
    true_pixels, n_prev, pcs, ccs):
    assert training_context is not None
    assert true_pixels is not None

    # Build error string
    dtype = data.dtype
    predictions = predictor.predict(training_context)
    estimated_pixels = convert_predictions_to_pixels(predictions, dtype)
    error_string = true_pixels - estimated_pixels

    if true_pixels.ndim == 2:
        to_reshape = (-1, true_pixels.shape[-1])
        residuals = np.empty((0, true_pixels.shape[-1]), dtype=dtype)
    else:
        to_reshape = (-1,)
        residuals = np.empty((0,), dtype=dtype)

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

    if true_pixels.ndim == 2:
        assert residuals.shape[0] + error_string.shape[0] == \
            np.prod(data.shape[:-1])
    else:
        assert residuals.shape[0] + error_string.shape[0] == \
            np.prod(data.shape)

    return (error_string, residuals, predictor), None, data.shape


def predictive_decomp(error_string, residuals, predictor, n_prev, pcs, ccs,
    original_shape):

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

    if error_string.ndim == 2:
        to_reshape = (original_shape[0]-n_prev, r_end-r_start, c_end-c_start, \
            error_string.shape[1])
    else:
        to_reshape = (original_shape[0]-n_prev, r_end-r_start, c_end-c_start)
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
                context = get_context(data, n_prev, pcs, ccs, n, r, c)
                prediction = predictor.predict(context)
                prediction = convert_predictions_to_pixels(prediction, dtype)
                data[n,r,c] = prediction + errors[n-n_prev, r-r_start,
                    c-c_start]

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
        exit()

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
        exit()

    return context.reshape((1, -1))
