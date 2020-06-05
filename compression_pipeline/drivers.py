from preprocessors.sqpatch import sqpatch_pre, sqpatch_post
from preprocessors.rgb import rgb_pre, rgb_post
from preprocessors.dct import dct_pre

from compressors.knn_mst import knn_mst_comp, knn_mst_decomp

from encoders.delta_coo import delta_coo_enc, delta_coo_dec
from encoders.delta_huffman import delta_huffman_enc, delta_huffman_dec
from encoders.video import video_enc


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
    '''

    preprocessor = args.pre

    if preprocessor == 'sqpatch':
        return sqpatch_pre(data, args.psz)
    elif preprocessor == 'rgb':
        return rgb_pre(data, args.rgbr, args.rgbc)
    elif preprocessor == 'rgb-sqpatch':
        rgb_data, _ = rgb_pre(data, args.rgbr, args.rgbc)
        return sqpatch_pre(rgb_data, args.psz)
    elif preprocessor == 'dct':
        return dct(data)


def compress(data, element_axis, args):
    '''
    Calls the appropriate compressor

    Args:
        data: numpy array
            data to be compressed
        element_axis: int
            index into data.shape for n_elements
        args: Namespace
            command-line argument namespace

    Returns:
        compressed_data: numpy array
            compressed data, of shape (n_layers, n_elements, n_points)
        metadata: numpy array
            metadata corresponding to each layer in the compression
        original_shape: tuple
            shape of original data
    '''

    compressor = args.comp

    if compressor == 'knn-mst':
        return knn_mst_comp(data, element_axis, args.n_neighbors,
            args.metric, args.minkowski_p)


def encode(compression, metadata, original_shape, args):
    '''
    Calls the appropriate encoder

    The encoder will write the compression to disk.

    Args:
        compression: numpy array
            compressed data to be encoded
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
        original_shape: tuple
            shape of original data
        args: Namespace
            command-line argument namespace

    Returns:
        None
    '''

    encoder = args.enc

    if encoder == 'delta-coo':
        delta_coo_enc(compression, metadata, original_shape, args)
    elif encoder == 'delta-huff':
        delta_huffman_enc(compression, metadata, original_shape, args)
    elif encoder == 'video':
        video_enc(compression, metadata, original_shape, args,
            args.video_codec, args.gop_strat, args.image_codec, args.framerate,
            args.grayscale)


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
        metadata: numpy array
            metadata is for compression (not necessarily the same metadata
            that is returned by the loader)
    '''

    decoder = args.enc

    if decoder == 'delta-coo':
        return delta_coo_dec(comp_file)
    elif decoder == 'delta-huff':
        return delta_huffman_dec(comp_file)


def decompress(compression, metadata, original_shape, args):
    '''
    Calls the appropriate decompressor

    Args:
        compression: numpy array
            compressed data
        metadata: numpy array
            metadata for compression (not necessarily the same metadata
            that is returned by the loader)
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
        return knn_mst_decomp(compression, metadata, original_shape)


def postprocess(decomp, args):
    '''
    Calls the appropriate postprocessor.

    Args:
        decomp: numpy array
            decompressed data
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
