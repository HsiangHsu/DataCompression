'''
preprocessors/dict.py

This module contains the Dictionary Learning preprocessor.
'''

import numpy as np
from sklearn.decomposition import DictionaryLearning, \
    MiniBatchDictionaryLearning

from datetime import timedelta
from timeit import default_timer as timer


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
        atoms: numpy array
            learned dictionary, of shape (n_components, n_points)
    '''

    data = data.reshape(data.shape[0], -1)

    start = timer()
    dico = MiniBatchDictionaryLearning(n_iter=10, n_components=10, random_state=1)
    dico.fit(data)
    atoms = dico.components_
    end = timer()
    print(f'\tlearn dictionary in {timedelta(seconds=end-start)}.')

    start = timer()
    transform = dico.transform(data).astype(np.uint16)
    end = timer()
    print(f'\ttransform data in {timedelta(seconds=end-start)}.\n')

    ###
    post_data = np.zeros((data.shape[0], atoms.shape[1]),
        dtype=np.float64)

    for i in range(post_data.shape[0]):
        post_data[i] = np.matmul(transform[i], atoms).astype(np.float64)

    from matplotlib import pyplot as plt

    plt.imshow(data[0].reshape((28,28)))
    plt.show()
    plt.imshow(post_data[0].astype(np.uint8).reshape((28,28)))
    plt.show()

    exit()
    np.save('udata_in', post_data)
    ###

    return transform, 0, atoms


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

    post_data = np.zeros((decomp.shape[0], pre_metadata.shape[1]),
        dtype=decomp.dtype)

    for i in range(post_data.shape[0]):
        post_data[i]  = np.matmul(decomp[i], pre_metadata).astype(post_data.dtype)

    return post_data
