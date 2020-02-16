import idx2numpy
import numpy as np

averages = [[[0 for i in range(28)] for i in range(28)] for i in range(10)]

for NUM in range(10):
    datapath = f'../datasets/mnist/train-images-quantized-{NUM}-idx3-ubyte'
    raw_data = idx2numpy.convert_from_file(datapath)

    for i in range(28):
        for j in range(28):
            counts = [0, 0]
            for datapoint in raw_data:
                if datapoint[i][j] == 0:
                    counts[0] += 1
                elif datapoint[i][j] == 1:
                    counts[1] += 1
                else:
                    assert("ERR")
            if counts[0] >= counts[1]:
                averages[NUM][i][j] = 0
            else:
                averages[NUM][i][j] = 1

averages_nparr = np.array(averages, dtype='uint8')
with open(f"../datasets/mnist/train-images-averages-idx3-ubyte",
    mode='wb') as out:
    idx2numpy.convert_to_file(out, averages_nparr)
