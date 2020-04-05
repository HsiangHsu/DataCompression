import idx2numpy
import numpy as np

datapath = '../datasets/mnist/train-images-idx3-ubyte'
labelpath = '../datasets/mnist/train-labels-idx1-ubyte'
raw_data = idx2numpy.convert_from_file(datapath)
raw_labels = idx2numpy.convert_from_file(labelpath)

for NUM in range(0, 10):
    quantized_data = []
    for label_idx in range(len(raw_labels)):
        if raw_labels[label_idx] != NUM:
            continue
        edited_datapoint = np.copy(raw_data[label_idx])
        for i in range(len(edited_datapoint)):
            for j in range(len(edited_datapoint[i])):
                if edited_datapoint[i][j] != 0:
                    edited_datapoint[i][j] = 1
        quantized_data.append(edited_datapoint)

    quantized_nparr = np.array(quantized_data)
    with open(f"../datasets/mnist/train-images-quantized-{NUM}-idx3-ubyte",
        mode='wb') as out:
        idx2numpy.convert_to_file(out, quantized_nparr)
