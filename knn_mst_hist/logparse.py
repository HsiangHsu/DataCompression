import numpy as np

mst_entropies = []

with open('logs/log_cifar_l0.txt', 'r') as f:
    color_i = -1
    for line in f:
        words = line.split()
        if words and words[0] == 'COLOR':
            mst_entropies.append([])
            color_i += 1
        if words and words[0] == 'Random':
            mst_entropies[color_i].append(float(words[2]))
            # mst_entropies.append(float(words[2]))

print(np.average(mst_entropies, axis=1))
# print(np.average(mst_entropies))
