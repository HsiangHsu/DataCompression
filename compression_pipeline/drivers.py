'''
drivers.py

This module contains helper functions for each stage of the compression
pipeline.
'''

from preprocessors.sqpatch import sqpatch_pre, sqpatch_post
from preprocessors.rgb import rgb_pre, rgb_post
from preprocessors.dict import dict_pre, dict_post
from preprocessors.predictive import train_predictor, name_to_context_pixels

from compressors.knn_mst import knn_mst_comp, knn_mst_decomp, DisconnectedKNN
from compressors.knn_mst import knn_mst_comp, knn_mst_decomp
from compressors.predictive import predictive_comp, predictive_decomp

from encoders.delta_coo import delta_coo_enc, delta_coo_dec
from encoders.delta_huffman import delta_huffman_enc, delta_huffman_dec
from encoders.predictive_huffman import pred_huffman_enc, pred_huffman_dec
from encoders.predictive_golomb import pred_golomb_enc, pred_golomb_dec
from encoders.predictive_huff_run import pred_huff_run_enc, pred_huff_run_dec
from encoders.video import video_enc

from numpy.random import default_rng
from numpy import log


def preprocess(data, args):
    '''
    Calls the appropriate preprocessor

    Args:
        data: numpy array
            data to be preprocessed
        args: Namespace
            command-line argument namespace

    Returns:
        preprocessed_data: numpy array
            preprocessed data
        element_axis: int
            index into data.shape for n_elements
        pre_metadata:
            metadata for preprocessing
    '''

    preprocessor = args.pre

    if preprocessor == 'sqpatch':
        return sqpatch_pre(data, args.psz)
    elif preprocessor == 'rgb':
        return rgb_pre(data, args.rgbr, args.rgbc)
    elif preprocessor == 'rgb-sqpatch':
        rgb_data, _, _ = rgb_pre(data, args.rgbr, args.rgbc)
        return sqpatch_pre(rgb_data, args.psz)
    elif preprocessor == 'dict':
        return dict_pre(data, args.nc, args.alpha, args.niter, args.bsz)
    elif preprocessor == 'predictive':
        ordered_data =  None
        n_elements = data.shape[0]
        if args.load_state is not None:
            h, m = args.load_state
            args.feature_file = f'training_context_{h}_{m}.npy'
            args.label_file = f'true_pixels_{h}_{m}.npy'
            args.predictor_file = f'predictor_{h}_{m}.out'
        if args.ordering == 'mst':
            alpha = 1
            while True:
                try:
                    # DOI: 0.1016/S0167-7152(02)00421-2 showed
                    # k = O(log n) usually
                    # gives connectedness in the k-NN.
                    k_val = alpha * int(log(n_elements))
                    # User specified a k value and we haven't tried it yet
                    if args.k and alpha == 1:
                        k_val = args.k
                    ordered_data, _, _ = knn_mst_comp(data, element_axis=0,
                        metric='euclidean', minkowski_p=2,
                        k=k_val)
                    break
                except DisconnectedKNN:
                    alpha *= 10
                    print("Disconnected kNN. Increasing k to",
                        alpha * int(log(n_elements)))
            ordered_data = ordered_data.reshape(n_elements, *data.shape[1:])
        elif args.ordering == 'hamiltonian':
            assert False, 'HAMILTONIAN ORDERING IS UNIMPLEMENTED'
        elif args.ordering == 'random':
            rng = default_rng()
            ordered_data = data[rng.permutation(n_elements)]


        should_extract = (args.feature_file is None)
        should_train = (args.predictor_file is None)

        cubist_rules_or_none = None if args.predictor_family != 'cubist' \
            else args.num_cubist_rules

        return train_predictor(args.predictor_family, ordered_data,
            args.num_prev_imgs, args.prev_context, args.curr_context,
            args.mode, cubist_rules_or_none,
            should_extract_training_pairs=should_extract,
            training_filenames=(args.feature_file, args.label_file),
            should_train=should_train,
            predictor_filename=args.predictor_file)


def compress(data, element_axis, pre_metadata, args):
    '''
    Calls the appropriate compressor

    Args:
        data: numpy array
            data to be compressed
        element_axis: int
            index into data.shape for n_elements
        pre_metadata:
            metadata generated by the preprocessor, used by some compressors
        args: Namespace
            command-line argument namespace

    Returns:
        compressed_data: numpy array
            compressed data, of shape (n_layers, n_elements, n_points)
        comp_metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
    '''

    compressor = args.comp

    if compressor == 'knn-mst':
        return knn_mst_comp(data, element_axis, args.metric, args.minkowski_p)
    elif compressor == 'predictive':
        return predictive_comp(data, *pre_metadata, args.mode)


def encode(compression, pre_metadata, comp_metadata, original_shape, args):
    '''
    Calls the appropriate encoder

    The encoder will write the compression to disk.

    Args:
        compression: numpy array
            compressed data to be encoded
        pre_metadata: numpy array
            metadata for preprocessing
        comp_metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
        args: Namespace
            command-line argument namespace

    Returns:
        None
    '''

    encoder = args.enc

    if encoder == 'delta-coo':
        delta_coo_enc(compression, comp_metadata, original_shape, args)
    elif encoder == 'delta-huff':
        delta_huffman_enc(compression, pre_metadata, comp_metadata,
            original_shape, args)
    elif encoder == 'video':
        video_enc(compression, metadata, original_shape, args,
            args.video_codec, args.gop_strat, args.image_codec, args.framerate,
            args.grayscale)
    elif encoder == 'pred-huff':
        pred_huffman_enc(compression, pre_metadata[-3:], original_shape, args)
    elif encoder == 'pred-golomb':
        pred_golomb_enc(compression, pre_metadata[-3:], original_shape, args)
    elif encoder == 'pred-huff-run':
        pred_huff_run_enc(compression, pre_metadata[-3:], original_shape, args)


def decode(comp_file, args):
    '''
    Calls the appropriate decoder

    Args:
        comp_file: string
            filepath to compressed data
        decoder: string
            decoder to use

    Returns:
        decompression: numpy array
            decompressed data
        pre_metadata: numpy array
            metadata for preprocessing
        comp_metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
    '''

    decoder = args.enc

    if decoder == 'delta-coo':
        return delta_coo_dec(comp_file)
    elif decoder == 'delta-huff':
        return delta_huffman_dec(comp_file)
    elif decoder == 'pred-huff':
        return pred_huffman_dec(comp_file)
    elif decoder == 'pred-golomb':
        return pred_golomb_dec(comp_file)
    elif decoder == 'pred-huff-run':
        return pred_huff_run_dec(comp_file)


def decompress(compression, comp_metadata, original_shape, args):
    '''
    Calls the appropriate decompressor

    Args:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression
        original_shape: tuple
            shape of original data
        decompressor: string
            decompressor to use

    Returns:
        decompression: numpy array
            decompressed data
    '''

    decompressor = args.comp

    if decompressor == 'knn-mst':
        return knn_mst_decomp(compression, comp_metadata, original_shape)
    elif decompressor == 'predictive':
        return predictive_decomp(*compression, *comp_metadata, original_shape,
                                 args.mode, args.predictor_family=='cubist')


def postprocess(decomp, pre_metadata, args):
    '''
    Calls the appropriate postprocessor.

    Args:
        decomp: numpy array
            decompressed data
        pre_metadata: numpy array
            metadata for preprocessing
        args: Namespace
            command-line argument namespace

    Returns:
        post_data: numpy array
            postprocessed data
    '''

    postprocessor = args.pre

    if postprocessor == 'sqpatch':
        return sqpatch_post(decomp)
    elif postprocessor == 'rgb':
        return rgb_post(decomp)
    elif postprocessor == 'rgb-sqpatch':
        return rgb_post(sqpatch_post(decomp))
    elif postprocessor == 'dict':
        return dict_post(decomp, pre_metadata)
    elif postprocessor == 'predictive':
        return decomp
