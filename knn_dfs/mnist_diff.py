# -*- coding: utf-8 -*-
""""
Codes for Dataset Compression
Authors: Hsiang Hsu (hsianghsu@g.harvard.edu), Alex Mariona, Madeleine Barowsky
"""
import argparse
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
from time import localtime, strftime
from sklearn.neighbors import NearestNeighbors
import cv2
from random import sample
from contextlib import redirect_stdout

from PIL import Image
import subprocess
import ffmpeg
import tempfile

from scipy.sparse.csgraph import minimum_spanning_tree
from mnist_mst_diff import *

############# COMMAND-LINE ARGUMENTS ########
parser = argparse.ArgumentParser("Compress image datasets with KNN MST and as video")
valid_intermediate_frame_codecs = ['jpg', 'png']
valid_output_video_codecs = ['av1', 'vp8', 'vp9']
parser.add_argument(
        '-video_codec', default='av1',
        help='video codec to be used for output (av1, vp8, vp9)')
parser.add_argument(
        '-image_codec', default='png',
        help='intermediate image frame codec (png, jpg)')
parser.add_argument(
        '-no_gop', action='store_true', default=False,
        help='whether the longest path in the MST should be used as the GoP parameter')
options = parser.parse_args()
if options.image_codec not in valid_intermediate_frame_codecs:
    parser.error('intermediate frame codec must be png or jpg')
if options.video_codec not in valid_output_video_codecs:
    parser.error('output video codec must be av1, vp8, or vp9')

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

def encode_video_from_imgs(video_extension, filename, imgs, shapes, FPS, longest_path, 
                           intermediate_file_format='png', fstream=None):
    # video_extension: string, either 'av1', 'vp8', or 'vp9'
    # filename: without the extension
    # imgs: numpy array of images
    # shapes: tuple of dimensions
    # FPS: int, frame per second
    # longest_path: int, number of images in the longest path in the MST or 
    #                    0 if the argument should not be passed to the encoder
    # intermediate_file_format: string, either 'jpg' or 'png'
    # fstream: stream, (default None) for logging
    assert longest_path >= 0
    assert intermediate_file_format in valid_intermediate_frame_codecs
    assert video_extension in valid_output_video_codecs

    intermediate_filename = 'intermediate_' + filename + '_output'
    optimize = True
    quality = 95
    if fstream:
        fstream.write(("Exporting frames as {0} images with " 
                       "optimize = {1} and quality = {2}\n").format(intermediate_file_format, 
                                                                    optimize, quality))

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(imgs.shape[0]):
            with Image.fromarray((255 * imgs[i].reshape(shapes))).convert('L') as im:
                # Quality param is for JPEG only but PNG will silently ignore
                im.save(os.path.join(tmpdirname, 'img' + str(i) + '.' + intermediate_file_format),
                        optimize=True, quality=95)
        
        # Generate intermediate mkv container in current directory for the ordered images
        ffmpeg.input(tmpdirname + '/*.' + intermediate_file_format, pattern_type='glob', framerate=str(FPS)
                    ).output(intermediate_filename + '.mkv', vcodec='copy').run()

    # -speed 0 is highest quality encoding but slowest
    args = ['-%s' % video_extension, '-speed', '0'] if video_extension != 'vp9' else ['-speed', '0']
    raw_ffmpeg_args = ''
    if video_extension != 'vp8':
        raw_ffmpeg_args += '-tile-columns 0 -tile-rows 0 '
        
    if longest_path > 0:
        raw_ffmpeg_args += '-g %s' % longest_path

    if raw_ffmpeg_args:
        args.append('-fo=%s' % raw_ffmpeg_args)
    
    if fstream:
        fstream.write("Encoding %s video with args = %r\n" % (video_extension, args))
        subprocess.run(['webm', '-i', intermediate_filename + '.mkv'] + args, check=True, stderr=fstream)
        # Rename final output file to the name that was requested
        subprocess.run(['mv', intermediate_filename + '.webm', filename + '.webm'], check=True, stdout=fstream, stderr=fstream)
    else:
        subprocess.run(['webm', '-i', intermediate_filename + '.mkv'] + args, check=True)
        # Rename final output file to the name that was requested
        subprocess.run(['mv', intermediate_filename + '.webm', filename + '.webm'], check=True)

def decode_imgs_from_video(video_filename, frame_codec='png', fstream=None):
    directory_name = ''.join(video_filename.split('.')[0:-1]) + '-frames'
    assert frame_codec in valid_intermediate_frame_codecs
    try:    
        os.mkdir(directory_name)
    except FileExistsError:
        print("Directory already exists")
    probe = ffmpeg.probe(video_filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    #pixel_fmt = video_stream['pix_fmt']
    ffmpeg.input(video_filename).output(directory_name + "/frame%d." + frame_codec, s='%dx%d' % (width, height), 
                                        start_number=1, pix_fmt='gray').run()
    


######### BEGIN LOGGING #########
filename = 'mnist_diff'
currenttime = localtime()
file = open(filename+'_log_'+strftime("%m-%d-%H.%M", currenttime)+'.txt','w')
log_current_timestamp(file, currenttime)

######### LOAD DATA #########
n_samples = 55000
mnist = tfds.load('mnist', split='train', as_supervised=True)
X = np.array(list(mnist.take(n_samples).as_numpy_iterator()))
X = np.array([elem[0].reshape(784,) for elem in X])
framerate = 24
video_format = options.video_codec

######### RANDOM ORDER FOR BENCHMARKING #########
file.write('Storing the datasets randomly as a %s video\n' % video_format)
gop_str = 'gop_' if not options.no_gop else 'nogop_'
encode_video_from_imgs(video_format, filename='mnist_random_'+options.image_codec+'_'+video_format+'_'+gop_str+str(n_samples), imgs=X, 
                       shapes=(28, 28), FPS=framerate, longest_path=0, 
                       intermediate_file_format=options.image_codec, fstream=file)
file.write('Finished encoding the random ordering %s video\n' % video_format)
log_current_timestamp(file)

######### K-NN AND MST #########
neighbors = 100
file.write('Fitting %d-NN graph\n' % neighbors)
# Fitting K-NN with Euclidean distance
with redirect_stdout(file):
    neigh = NearestNeighbors(n_neighbors=neighbors, metric='minkowski', p=2).fit(X)
file.write('Finished fitting %d-NN graph\n' % neighbors)
log_current_timestamp(file)

# AGM: Use minimum spanning tree to find a stream of images.
file.write('Finding minimum spanning tree\n')
csr = None
with redirect_stdout(file):
    csr = minimum_spanning_tree(neigh.kneighbors_graph(mode='distance')).toarray()
file.write('Finished finding minimum spanning tree\n')
log_current_timestamp(file)
edges = csr_to_edges(csr)
tree = create_tree(edges)
file.write('Converting MST into image ordering\n')
nodes, longest_path_len = mst_to_order(tree, edges, return_longest_path_len=True)
file.write('Finished converting MST into image ordering\n')
log_current_timestamp(file)

file.write('Storing the datasets using our algorithm as %s video\n' % video_format)
encode_video_from_imgs(video_format, filename='mnist_diff_'+options.image_codec+'_'+video_format+'_'+gop_str+str(n_samples), imgs=X[nodes],
                       shapes=(28, 28), FPS=framerate, longest_path=(0 if options.no_gop else longest_path_len), 
                       intermediate_file_format=options.image_codec, fstream=file)
file.write('Finished encoding the ordered dataset as %s video\n' % video_format)
log_current_timestamp(file)

######### Serialize the ordered images from our algorithm #########
file.write('Saving Results\n')
file.flush()
with open('mnist_' + str(n_samples) + '_diff_' + strftime("%m-%d-%H-%M", currenttime) + '.pickle', 'wb') as f:
    save = {
        'idx': nodes,
        'imgs': X[nodes]
        }
    pickle.dump(save, f, 2)

file.close()
