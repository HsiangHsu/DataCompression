'''
preprocessors/dict.py

This module contains the Dictionary Learning preprocessor.
'''

import numpy as np
from sklearn.decomposition import DictionaryLearning, \
    MiniBatchDictionaryLearning, dict_learning_online, dict_learning

from datetime import timedelta
from timeit import default_timer as timer

from matplotlib import pyplot as plt


def dict_pre(data, n_components, alpha, n_iter, batch_size):
    '''
    Dictionary Learning Preprocessor

    Uses Mini-Batch Dictionary Learning to generate a dictionary of atoms to
    sparsely represent the data.

    Args:
        data: numpy array
            data to be preprocessed, of shape (n_elements, n_points)

    Returns:
        code: numpy array
            transformed data using learned dictionary, of shape
            (n_elements, n_components)
        element_axis: int
            index into code.shape for n_elements
        dictionary: numpy array
            learned dictionary, of shape (n_components, n_points)
    '''

    # TODO: Send original datatype?

    data = data.reshape(data.shape[0], -1)

    start = timer()
    code, dictionary = dict_learning_online(data, n_components=n_components,
        alpha=alpha, n_iter=n_iter, batch_size=batch_size, method='cd',
        positive_dict=True, positive_code=True, return_code=True,
        random_state=0)
    code = code.astype(np.uint32)
    end = timer()
    np.save('code', code)
    np.save('dictionary', dictionary)
    print(f'\tlearn dictionary in {timedelta(seconds=end-start)}.\n')

    expected_data = np.matmul(code, dictionary).astype(np.uint8)
    np.save('udata_in', expected_data)

    return code, 0, dictionary


def dict_post(decomp, pre_metadata):
    '''
    Dictionary Learning Postprocessor

    See docstring on the corresponding preprocessor for more information

    Args:
        decomp: numpy array
            decompressed data
        pre_metadata: numpy array
            learned dictionary

    Returns:
        post_data: numpy array
            postprocessed data, of shape
    '''

    # TODO: Universal original datatype?

    return np.matmul(decomp, pre_metadata).astype(np.uint8)
