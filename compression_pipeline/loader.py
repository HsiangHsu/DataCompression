'''
loader.py

This the module responsible for unpacking the desired dataset into a properly
shaped numpy array.
'''


import idx2numpy
import numpy as np
import os
import pandas as pd
import pickle


def load_dataset(dataset):
    '''
    Loads and shapes a dataset into a numpy array

    All datasets should be stored in /DataCompression/datasets/{dataset}.

    What the 'proper' shape of the numpy array may depend on the method of
    compression. Currently, all data is image data, so the proper shape is
    (n_elements, [n_layers], width, height). n_layers refers to number of
    channels, and may not be present if the image is greyscale.
    For example, MNIST is (60000, 28, 28) and CIFAR-10 is (50000, 3, 32, 32).

    The datatype of the numpy array should be specified if it is being built
    from scratch.

    Processing metadata is currently a TODO, as it is not handled by the
    rest of the pipeline. As such, care is not taken to properly format
    (or even return) metadata in all cases.

    Args:
        dataset: string
            name of dataset, passed as a command line argument to
            driver_compress.py

    Returns:
        (data, meta): (numpy array, numpy array?) tuple
            tuple containing loaded and shaped data and metadata, such as
            labels or filenames
    '''

    dirname = os.path.dirname(__file__)
    dirpath = os.path.join(dirname, f'../datasets/{dataset}')

    if dataset == 'test':
        datapath = 'test_data.np'
        with open(os.path.join(dirpath, datapath), 'rb') as f:
            data = np.load(f)
            return (data, None)

    elif dataset == 'synthetic':
        datapath = 'synthetic_data.np'
        with open(os.path.join(dirpath, datapath), 'rb') as f:
            data = np.load(f)
            return (data, None)

    elif dataset == 'mnist':
        datapath = 'train-images-idx3-ubyte'
        labelpath = 'train-labels-idx1-ubyte'
        data = idx2numpy.convert_from_file(os.path.join(dirpath, datapath))
        labels = idx2numpy.convert_from_file(
            os.path.join(dirpath, labelpath))
        return (data, labels)

    elif dataset == 'cifar-10':
        data = np.empty((50000, 3072), dtype=np.uint8)
        for i in range(1,6):
            datapath = f'data_batch_{i}'
            with open(os.path.join(dirpath, datapath), 'rb') as f:
                raw_data = pickle.load(f, encoding='bytes')
                data[(i-1)*10000:i*10000] = raw_data[b'data']
        data = data.reshape(50000, 3, 32, 32)
        return (data, None)

    elif dataset == 'adult':
        datapath = 'adult.data'
        df = pd.read_csv(os.path.join(dirpath, datapath), header=None,
            sep=r',\s+', engine='python')
        df.columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "gender",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"
        ]
        discrete_cols = df.select_dtypes(['object']).columns
        for col in discrete_cols:
            df[col] = df[col].astype('category').cat.codes
        for col in df:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
        return (np.ascontiguousarray(df.to_numpy()), None)
