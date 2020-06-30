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


def dict_pre(data):
    '''
    Dictionary Learning preprocessor

    Uses Mini-Batch Dictionary Learning to generate a dictionary of atoms to
    sparsely represent the data.

    Args:
        data: numpy array
            data to be preprocessed, of shape (n_elements, n_points)

    Returns:
        transform: numpy array
            transformed data using learned dictionary, of shape
            (n_elements, n_components)
        element_axis: int
            index into transform.shape for n_elements
        dictionary: numpy array
            learned dictionary, of shape (n_components, n_points)
    '''

    data = data.reshape(data.shape[0], -1)

    start = timer()
    code, dictionary = dict_learning_online(data, n_components=data.shape[1],
        n_iter=25, return_code=True, batch_size=10, random_state=1,
        dict_init=None, positive_dict=True, positive_code=True, method='cd',
        verbose=0, method_max_iter=1000)
    code = code.astype(np.uint32)
    end = timer()
    np.save('code', code)
    np.save('dictionary', dictionary)
    print(f'\n\tlearn dictionary in {timedelta(seconds=end-start)}.\n')

    expected_data = np.matmul(code, dictionary).astype(np.uint8)
    np.save('udata_in', expected_data)

    # expected_data = expected_data.reshape((data.shape[0],28,28))
    # data = data.reshape((data.shape[0],28,28))
    # for i in range(len(expected_data)):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(expected_data[i])
    # plt.show()

    return code, 0, dictionary


def dict_post(decomp, pre_metadata):
    '''
    RGB reshaping postprocessor

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

    post_data = np.matmul(decomp, pre_metadata).astype(np.uint8)

    # data = post_data.reshape((post_data.shape[0],28,28))
    # for i in range(len(data)):
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(data[i])
    # plt.show()

    return post_data
