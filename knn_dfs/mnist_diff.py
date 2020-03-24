# -*- coding: utf-8 -*-
""""
Codes for Dataset Compression
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import tensorflow as tf
import numpy as np
import pickle
from time import localtime, strftime
from sklearn.neighbors import NearestNeighbors
import cv2
from random import sample

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

filename = 'mnist_diff'
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

# n_samples = mnist.train.images.shape[0]
# X = mnist.train.images

n_samples = 10000
idx = np.random.choice(mnist.train.labels.shape[0], n_samples, replace=False)
X = mnist.train.images[idx, :]

def video_maker(filename, imgs, shapes, FPS):
    # filename: string, e.g., xxx.avi
    # imgs: numpy array of images
    # sahpes: tuple of dimensions
    # FPS: int, frame per second

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=float(FPS), frameSize=shapes)
    for i in range(imgs.shape[0]):
        frame = cv2.cvtColor(imgs[i, :].reshape(shapes)*255.0, cv2.COLOR_GRAY2RGB).astype('uint8')
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
    return

file.write('Storing the datasets randomly as a video\n')
video_maker(filename='mnist_random_'+str(n_samples)+'.avi', imgs=X, shapes=(28, 28), FPS=24)
file.flush()

file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()
# Fitting k means
neigh = NearestNeighbors(n_neighbors=100, radius=1.0, metric='minkowski', p=2).fit(X)

file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()
# A greedy algorithm to find a stream of images
nodes = []
i = 0
diff_imgs = np.zeros((n_samples, 784))
starting = np.random.randint(0, n_samples, 1)

nodes.append(starting[0])
diff_imgs[i, :] = X[starting, :]

while len(nodes) != n_samples:

    neigh_dist, neigh_ind = neigh.kneighbors(X[starting, :].reshape(1, -1))
    neigh_ind = np.squeeze(neigh_ind)
    for j in range(len(nodes)):
        neigh_ind = np.delete(neigh_ind, np.where(neigh_ind==nodes[j]))
    if len(neigh_ind) != 0:
        nodes.append(neigh_ind[0])
        starting = neigh_ind[0]
        diff_imgs[i, :] = X[starting, :]
    else:
        all_ = np.arange(n_samples).tolist()
        candidates = [a for a in all_ if a not in nodes]
        starting = sample(candidates, 1)
        nodes.append(starting[0])
        diff_imgs[i, :] = X[starting, :]

    if len(nodes) % 200 == 0:
        file.write('{}/{}\n'.format(len(nodes), n_samples))
        file.flush()

    i=i+1

file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

file.write('Storing the datasets using our algorithm as a video\n')
video_maker(filename='mnist_diff_'+str(n_samples)+'.avi', imgs=X[nodes], shapes=(28, 28), FPS=24)
file.flush()

file.write('Saving Results\n')
file.flush()
f = open('mnist_diff.pickle', 'wb')
save = {
    'idx': nodes,
    'imgs': X[nodes]
    }
pickle.dump(save, f, 2)
f.close()

file.close()
