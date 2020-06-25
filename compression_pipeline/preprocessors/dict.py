'''
preprocessors/dict.py

This module contains the Dictionary Learning preprocessor.
'''

from sklearn.decomposition import DictionaryLearning, \
    MiniBatchDictionaryLearning


def dict_pre(data):
    '''
    Dictionary Learning preprocessor

    DESC

    Args:
        data: numpy array
            data to be preprocessed, of shape
            (n_elements, 3*rows*cols)

    Returns:
        rgb_data: numpy array
            RGB data, of shape
            (3, n_elements, rows, cols)
        element_axis: int
            index into data.shape for n_elements
    '''

    data = data.reshape(data.shape[0], -1)
    print(data.shape)

    dico = DictionaryLearning(verbose=True, n_jobs=-1)
    V = dico.fit(data).components_
    print(V.shape)
    exit()

    return V, -1


def rgb_post(decomp):
    '''
    RGB reshaping postprocessor

    See docstring on the corresponding preprocessor for more information

    Args:
        decomp: numpy array
            decompressed data, of shape
            (3, n_elements, rows, cols)

    Returns:
        post_data: numpy array
            postprocessed data, of shape
            (n_elements, 3*rows*cols)
    '''

    assert decomp.shape[0] == 3, f'invalid shape for RGB data: {decomp.shape}'

    post_data = decomp.swapaxes(0, 1)
    n_elements = post_data.shape[0]
    post_data = post_data.reshape(n_elements, np.prod(post_data.shape[1:]))

    return post_data
