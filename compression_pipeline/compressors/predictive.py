'''
predictive.py

This module contains helper functions for implementing the predictive coding
compressor.
'''
import numpy as np
from utilities import name_to_context_pixels, convert_predictions_to_pixels, get_valid_pixels_for_predictions

def predictive_comp(data, element_axis, predictor, training_context,
    true_pixels, n_prev, pcs, ccs):
    assert training_context is not None
    assert true_pixels is not None

    # Build error string
    dtype = data.dtype
    error_string = np.array([], dtype=dtype)
    remaining_samples_to_predict = len(true_pixels)
    start_index = 0
    while remaining_samples_to_predict > 0:
        predict_batch_size = min(remaining_samples_to_predict, 1000)
        dtype = training_context.dtype
        estimated_pixels = convert_predictions_to_pixels(predictor.predict(
            training_context[start_index:start_index + predict_batch_size]), dtype)
        error_string = np.append(error_string,
                                 estimated_pixels - true_pixels[start_index:start_index + predict_batch_size])
        start_index += predict_batch_size
        remaining_samples_to_predict -= predict_batch_size

    # Build residuals
    residuals = np.array([], dtype=dtype)
    residuals = np.append(residuals, data[:n_prev].flatten())
    current_context_indices = name_to_context_pixels(ccs)
    prev_context_indices = name_to_context_pixels(pcs)
    r_start, r_end, c_start, c_end = get_valid_pixels_for_predictions(data[0].shape, 
                                        current_context_indices, prev_context_indices)
    r_end += 1
    c_end += 1
    for img in data[n_prev:]:
        residuals = np.append(residuals, img[:r_start,:])
        residuals = np.append(residuals, img[r_start:,:c_start])
        residuals = np.append(residuals, img[r_end:,c_start:])
        residuals = np.append(residuals, img[r_start:r_end,c_end:])
    assert residuals.shape[0] + error_string.shape[0] == np.prod(data.shape)

    return (error_string, residuals, predictor), None, data.shape


def predictive_decomp(error_string, residuals, predictor, n_prev, pcs, ccs,
    original_shape):

    dtype = error_string.dtype

    data = np.empty(original_shape, dtype=dtype)
    current_context_indices = name_to_context_pixels(ccs)
    prev_context_indices = name_to_context_pixels(pcs)
    r_start, r_end, c_start, c_end = get_valid_pixels_for_predictions(original_shape[1:], 
                                        current_context_indices, prev_context_indices)
    r_end += 1
    c_end += 1
    errors = error_string.reshape((original_shape[0]-n_prev, r_end-r_start,
        c_end-c_start))

    for n in range(n_prev):
        to_pop = original_shape[1] * original_shape[2]
        data[n], residuals = residuals[:to_pop].reshape(original_shape[1],
            original_shape[2]), residuals[to_pop:]

    for n in range(n_prev, original_shape[0]):
        data, residuals = load_residuals(data, residuals, original_shape, n,
            r_start, r_end, c_start, c_end)

        # Run predictor over remaining pixels
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                context = get_context(data, n_prev, pcs, ccs, n, r, c)
                prediction = convert_predictions_to_pixels(predictor.predict(context), dtype)
                data[n,r,c] = prediction + errors[n-n_prev, r-r_start,
                    c-c_start]

    assert len(residuals) == 0, \
        f'Not all residuals were consumed: {len(residuals)} pixels leftover.'

    return data


def load_residuals(data, residuals, original_shape, n, r_start, r_end, c_start,
    c_end):
    nr = r_start
    nc = original_shape[2]
    to_pop = nr*nc
    data[n,:r_start,:], residuals = \
        residuals[:to_pop].reshape((nr, nc)), residuals[to_pop:]

    nr = original_shape[1] - r_start
    nc = c_start
    to_pop = nr*nc
    data[n,r_start:,:c_start], residuals = \
        residuals[:to_pop].reshape((nr, nc)), residuals[to_pop:]

    nr = original_shape[1] - r_end
    nc = original_shape[2] - c_start
    to_pop = nr*nc
    data[n,r_end:,c_start:], residuals = \
        residuals[:to_pop].reshape((nr, nc)), residuals[to_pop:]

    nr = r_end - r_start
    nc = original_shape[2] - c_end
    to_pop = nr*nc
    data[n,r_start:r_end,c_end:], residuals = \
        residuals[:to_pop].reshape((nr, nc)), residuals[to_pop:]

    return data, residuals


def get_context(data, n_prev, pcs, ccs, n, r, c):
    context = np.empty((1, len(ccs)+len(pcs)*n_prev))
    if ccs == 'DAB':
        context[0, 0] = data[n,r,c-1]
        context[0, 1] = data[n,r-1,c-1]
        context[0, 2] = data[n,r-1,c]
    elif ccs == 'DABC':
        context[0, 0] = data[n,r,c-1]
        context[0, 1] = data[n,r-1,c-1]
        context[0, 2] = data[n,r-1,c]
        context[0, 3] = data[n,r-1,c+1]
    else:
        print(f'Current context string {ccs} unsupported by decompressor.')
        exit()

    if pcs == 'DAB':
        for p in range(n_prev):
            context[0, len(ccs)+3*p] = data[n-(p+1),r,c-1]
            context[0, len(ccs)+3*p+1] = data[n-(p+1),r-1,c-1]
            context[0, len(ccs)+3*p+2] = data[n-(p+1),r-1,c]
    elif pcs == 'DABC':
        for p in range(n_prev):
            context[0, len(ccs)+4*p] = data[n-(p+1),r,c-1]
            context[0, len(ccs)+4*p+1] = data[n-(p+1),r-1,c-1]
            context[0, len(ccs)+4*p+2] = data[n-(p+1),r-1,c]
            context[0, len(ccs)+4*p+3] = data[n-(p+1),r-1,c+1]
    else:
        print(f'Previous context string {pcs} unsupported by decompressor.')
        exit()

    return context
