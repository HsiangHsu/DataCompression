# -*- coding: utf-8 -*-
""""
Codes for Dataset Compression
Authors: Hsiang Hsu (hsianghsu@g.harvard.edu), Alex Mariona, Madeleine Barowsky
"""
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pickle
from time import localtime, strftime
from sklearn.neighbors import NearestNeighbors
import cv2
from random import sample

from PIL import Image
import subprocess
import ffmpeg

from scipy.sparse.csgraph import minimum_spanning_tree
from mnist_mst_diff import *

# I think this is deprecated and tensorflow_datasets.load is preferred...but anyway
from tensorflow.examples.tutorials.mnist import input_data

# Given an open filesteam |f|, logs |timestamp| (defualts to localtime())
# and flushes without closing
def log_current_timestamp(f, timestamp=None):
    if timestamp is None:
        timestamp = localtime()
    f.write(strftime("%Y-%m-%d-%H.%M.%S\n", timestamp))
    f.flush()

def avi_video_maker(filename, imgs, shapes, FPS):
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

def encode_av1_from_imgs(filename, imgs, shapes, FPS, longest_path):
    # filename: without the extension
    # imgs: numpy array of images
    # shapes: tuple of dimensions
    # FPS: int, frame per second
    # longest_path: int, number of images in the longest path in the MST or 
    #                    0 if the argument should not be passed to the encoder
    assert longest_path >= 0

    for i in range(imgs.shape[0]):
        with Image.fromarray((255 * imgs[i].reshape(shapes))).convert('L') as im:
            im.save('./tmp/img'+str(i)+'.jpg')
    
    intermediate_filename = 'intermediate_'+filename+'_output'
    # Generate intermediate mkv container for the ordered images
    ffmpeg.input('tmp/*.jpg', pattern_type='glob', framerate=str(FPS)
                ).output(intermediate_filename + '.mkv', vcodec='copy').run()

    # TODO incorporate both these
    args = ['--passes=2']
    if longest_path > 0:
        args.append('--kf-max-dist=' + str(longest_path)) 
    
    subprocess.check_call(['webm', '-i', intermediate_filename + '.mkv', '-av1'])
    subprocess.check_call(['mv', intermediate_filename + '.webm', filename + '.webm'])

    return


######### BEGIN LOGGING #########
filename = 'mnist_diff'
currenttime = localtime()
file = open(filename+'_log_'+strftime("%m-%d-%H.%M\n", currenttime)+'.txt','w')
log_current_timestamp(file, currenttime)

######### LOAD DATA #########
mnist = input_data.read_data_sets('MNIST_data')
n_samples = 10000
idx = np.random.choice(mnist.train.labels.shape[0], n_samples, replace=False)
X = mnist.train.images[idx, :]

######### RANDOM ORDER FOR BENCHMARKING #########
file.write('Storing the datasets randomly as a VP8 video\n')
video_maker(filename='mnist_random_'+str(n_samples)+'.avi', imgs=X, shapes=(28, 28), FPS=24)
file.flush()

######### K-MEANS AND MST #########
log_current_timestamp(file)
# Fitting k means
neigh = NearestNeighbors(n_neighbors=100, radius=1.0, metric='minkowski', p=2).fit(X)
log_current_timestamp(file)

# AGM: Use minimum spanning tree to find a stream of images.
csr = minimum_spanning_tree(neigh.kneighbors_graph(mode='distance')).toarray()
edges = csr_to_edges(csr)
tree = create_tree(edges)
nodes = mst_to_order(tree, edges)
log_current_timestamp(file)

file.write('Storing the datasets using our algorithm as a video\n')
video_maker(filename='mnist_diff_'+str(n_samples)+'.avi', imgs=X[nodes], shapes=(28, 28), FPS=24)
file.flush()

######### NOT SURE WHAT THIS IS #########
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
