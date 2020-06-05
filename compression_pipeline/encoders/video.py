'''
encoders/video.py

This module contains the Video encoder
'''

import ffmpeg
import pickle
from PIL import Image
import subprocess
import tempfile
import os


def video_enc(compression, metadata, original_shape, args, video_codec,
    gop_strat, image_codec, framerate, grayscale):
    '''
    Args:
        compression: numpy array
            compressed data to be encoded, of shape
            (n_layers, n_elements, n_points)
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader), of shape
            (n_layers, n_elements); since this encoder is intended
            to be used with smart orderings, this will probably be inverse
            orderings of some sort
        original_shape: tuple
            shape of original data
        args: dict
            all command line arguments passed to driver_compress.py
        video_codec: string
            either 'av1', 'vp8', or 'vp9'
        gop_strat: string
            one of 'default', 'max' or [INT] to specify the value for max GoP
            length; 'max' indicates that the number of frames in the dataset
            will be encoded relative to the first frame as a keyframe
        image_codec: string
            file format for intermediate frames, either 'jpg' or 'png'
        framerate: int
            frames per second of the final video
        grayscale: boolean
            whether the image data is grayscale
    '''
    n_layers = compression.shape[0]
    n_elements = compression.shape[1]
    n_points = compression.shape[2]

    ordered_imgs = compression.reshape((n_layers * n_elements, n_points))
    gop_len = 0
    if gop_strat == 'max':
        gop_len = -1
    elif gop_strat != 'default':
        gop_len = int(gop_strat)
    encode_video_from_imgs(video_codec, filename='comp.out', imgs=ordered_imgs,
                           shapes=original_shape[1:3], FPS=framerate,
                           gop_len=gop_len,
                           intermediate_file_format=image_codec,
                           grayscale=grayscale)
    with open('args.out', 'wb') as f:
        pickle.dump(args, f)


def encode_video_from_imgs(video_extension, filename, imgs, shapes, FPS, gop_len,
                           intermediate_file_format='png', grayscale=False):
    # video_extension: string, either 'av1', 'vp8', or 'vp9'
    # filename: output filename without the extension
    # imgs: numpy array of images (num_images, shapes[0] * shapes[1])
    # shapes: tuple of dimensions for a frame
    # FPS: int, frame per second
    # gop_len: int, 0 if the argument should not be passed to the encoder or -1
    #          if the number of frames should be used
    # intermediate_file_format: string, either 'jpg' or 'png'
    # grayscale: if the images are grayscale; specifying this should improve compression
    assert gop_len >= -1
    assert imgs[0].shape[0] == shapes[0] * shapes[1]

    intermediate_filename = 'intermediate_output'
    optimize = True
    quality = 95
    num_images = imgs.shape[0]

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(num_images):
            if grayscale:
                with Image.fromarray((255 * imgs[i].reshape(shapes))).convert('L') as im:
                    # Quality param is for JPEG only but PNG will silently ignore
                    im.save(os.path.join(tmpdirname, 'img' + str(i) + '.' + intermediate_file_format),
                            optimize=True, quality=95)
            else:
                with Image.fromarray((255 * imgs[i].reshape(shapes))) as im:
                    # Quality param is for JPEG only but PNG will silently ignore
                    im.save(os.path.join(tmpdirname, 'img' + str(i) + '.' + intermediate_file_format),
                            optimize=True, quality=95)

        # Generate intermediate mkv container in current directory for the ordered images
        ffmpeg.input(tmpdirname + '/*.' + intermediate_file_format, pattern_type='glob', framerate=str(FPS)
                    ).output(intermediate_filename + '.mkv', vcodec='copy').global_args('-loglevel', 'error').run()

    # -speed 0 is highest quality encoding but slowest
    args = ['-%s' % video_extension, '-speed', '0'] if video_extension != 'vp9' else ['-speed', '0']
    raw_ffmpeg_args = '-hide_banner -loglevel error '
    if video_extension != 'vp8':
        raw_ffmpeg_args += '-tile-columns 0 -tile-rows 0 '

    if gop_len == -1:
        gop_len = num_images

    if gop_len > 0:
        raw_ffmpeg_args += '-g %s' % gop_len

    if raw_ffmpeg_args:
        args.append('-fo=%s' % raw_ffmpeg_args)

    subprocess.run(['webm', '-i', intermediate_filename + '.mkv'] + args, check=True)
    # Rename final output file to the name that was requested
    subprocess.run(['mv', intermediate_filename + '.webm', filename + '.webm'], check=True)
    subprocess.run(['rm', intermediate_filename + '.mkv'], check=True)
