import os

def get_dataset_path(dataset):
    dirname = os.path.dirname(__file__)
    dirpath = os.path.join(dirname, f'../datasets/{dataset}')

    if dataset == 'test':
        return [os.path.join(dirpath, 'test_data.np')]

    elif dataset == 'synthetic':
        return [os.path.join(dirpath, 'synthetic_data.np')]

    elif dataset == 'mnist':
        datapaths = []
        datapaths.append(os.path.join(dirpath, 'train-images-idx3-ubyte'))
        datapaths.append(os.path.join(dirpath, 'train-labels-idx1-ubyte'))
        return datapaths

    elif dataset == 'cifar-10':
        datapaths = []
        for i in range(1,6):
            datapath = f'data_batch_{i}'
            datapaths.append(os.path.join(dirpath, datapath))
        datapaths.append(os.path.join(dirpath, 'test_batch'))
        datapaths.append(os.path.join(dirpath, 'batches.meta'))
        return datapaths
