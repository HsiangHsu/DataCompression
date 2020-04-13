import numpy as np

mst_entropies = []

with open('log_cifar_l2_2.txt', 'r') as f:
    color_i = -1
    for line in f:
        words = line.split()
        if words and words[0] == 'COLOR':
            mst_entropies.append([])
            color_i += 1
        if words and words[0] == 'MST':
            mst_entropies[color_i].append(float(words[2]))

print(np.average(mst_entropies, axis=1))
