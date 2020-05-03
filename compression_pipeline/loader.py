import idx2numpy
import numpy as np
import os

def load_dataset(dataset):
    dirname = os.path.dirname(__file__)
    dirpath = os.path.join(dirname, f'../datasets/{dataset}')

    if dataset == 'test':
        datapath = 'test_data.np'
        with open(os.path.join(dirpath, datapath), 'rb') as f:
            data = np.load(f)
            return (data, None)

    if dataset == 'mnist':
        datapath = 'train-images-idx3-ubyte'
        labelpath = 'train-labels-idx1-ubyte'
        data = idx2numpy.convert_from_file(os.path.join(dirpath, datapath))
        labels = idx2numpy.convert_from_file(os.path.join(dirpath, labelpath))
        return (data, labels)

load_dataset('mnist')
