import idx2numpy
import numpy as np
import os
import pandas as pd
import pickle


def load(dataset):
    dirname = os.path.dirname(__file__)
    dirpath = os.path.join(dirname, f'../datasets/{dataset}')

    if dataset == 'mnist':
        datapath = 'train-images-idx3-ubyte'
        labelpath = 'train-labels-idx1-ubyte'
        data = idx2numpy.convert_from_file(os.path.join(dirpath, datapath))
        return data

    elif dataset == 'cifar-10':
        data = np.empty((50000, 3072), dtype=np.uint8)
        for i in range(1,6):
            datapath = f'data_batch_{i}'
            with open(os.path.join(dirpath, datapath), 'rb') as f:
                raw_data = pickle.load(f, encoding='bytes')
                data[(i-1)*10000:i*10000] = raw_data[b'data']
        data = data.reshape((-1, 3, 32, 32)).transpose((0,2,3,1))
        return data
