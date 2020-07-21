'''
preprocessors/predictive.py

This module contains the Predictive Coding preprocessor.
'''
import numpy as np
from sklearn import linear_model
from scipy.sparse import csr_matrix

from datetime import timedelta, datetime
from timeit import default_timer as timer

import pickle

from utilities import name_to_context_pixels, convert_predictions_to_pixels, get_valid_pixels_for_predictions

def line_order_raster_image_to_1d(img):
        """
        A B C
        D E F    -->    A B C D E F G H I
        G H I
        """
        return img.flatten()

def zig_zag_raster_image_to_1d(img):
        """
        A B C
        D E F    -->    A B D G E C F H I
        G H I
        """
        scan = np.zeros((img.shape[0] * img.shape[1],), dtype=img.dtype)
        l_to_r = True
        total_diagonals = img.shape[0] + img.shape[1] - 1
        i = 0
        for diag_index in range(total_diagonals):
                if diag_index <= total_diagonals // 2:
                        row_iter = range(diag_index, -1, -1) if l_to_r else range(0, diag_index + 1)
                        for row in row_iter:
                                scan[i] = img[row][diag_index - row]
                                i += 1
                else:
                        # below the diagnoal
                        row_iter = range(img.shape[0] - 1, diag_index - img.shape[0], -1) if l_to_r else range(diag_index - img.shape[0] + 1, img.shape[0])
                        for row in row_iter:
                                scan[i] = img[row][diag_index - row]
                                i += 1
                l_to_r = (not l_to_r)
        return scan

def hilbert_scan_raster_image_to_1d(img):
        """
        A B C D
        E F G H  -->    A E F B C G H D I M N J K O P L (maybe ???)
        I J K L
        M N O P
        """
        print('Hilbert scan is unimplemented')
        pass


def apply_relative_indices(relative_indices, i, j):
        '''
        Returns in a format suitable for ndarray indexing
        '''
        assert i >= 0
        assert j >= 0
        return ([x + i for (x, y) in relative_indices], [y + j for (x, y) in relative_indices])


def extract_training_pairs(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
    '''
    Assumes |prev_context_indices| and |current_context_indices| are lists of relative indices on (i, j)
    '''
    pixels_to_predict = get_valid_pixels_for_predictions(ordered_dataset[0].shape,
                                                         current_context_indices, prev_context_indices,
                                                         return_tuples=True)
    X_train = []
    Y_train = []
    for current_img_index in range(num_prev_imgs, ordered_dataset.shape[0]):
        for (i, j) in pixels_to_predict:
            predictive_string = []
            predictive_string.append(ordered_dataset[current_img_index][apply_relative_indices(current_context_indices, i, j)])
            for a in range(num_prev_imgs):
                predictive_string.append(ordered_dataset[current_img_index - a - 1][apply_relative_indices(prev_context_indices, i, j)])
            X_train.append(np.concatenate(predictive_string))
            Y_train.append(ordered_dataset[current_img_index][i][j])
    return (X_train, Y_train)


def __compute_classifier_accuracy(clf, predictor_family, training_context, true_pixels):
    if predictor_family == 'logistic' or predictor_family == 'linear':
        n_samples_float = 1.0 * len(true_pixels)
        predicted_pixels = convert_predictions_to_pixels(clf.predict(training_context), training_context.dtype)
        return 1 - np.count_nonzero(predicted_pixels - true_pixels) / n_samples_float
    assert False, 'Must be a logistic or linear predictor to compute accuracy'


def train_predictor(predictor_family, ordered_dataset, num_prev_imgs, prev_context, cur_context,
                    should_extract_training_pairs=True, training_filenames=None):
    '''
    Generalized predictive coding preprocessor

    Args:
        predictor_family: str
            one of 'linear', 'logistic' for the regression family to compute across training features and labels
        ordered_dataset: numpy array
            data to be preprocessed, of shape (n_elements, n_points)
        num_prev_imgs: int
            how many images preceeding each element should be considered as
            "dataset context"
        prev_context: string
            string describing the relative location of a pixel to be used
            for context in each "dataset context" image
        cur_context: string
            string describing the relative location of a pixel to be used
            for context in the current image
        should_extract_training_pairs: boolean (default True)
            whether to compute the training context from the dataset or else
            load from |training_pairs_filename|
        training_filenames: tuple of (string, string) (default None)
            if |should_extract_training_pairs|, the locations of the NumPy
            files from which to load |training_context| and |true_pixels|,
            respectively

    Returns:
        ordered_dataset: numpy array
            unchanged from input
        element_axis: int
            index into ordered_dataset.shape for n_elements
        (clf, training_context, true_pixels): tuple of
            (sklearn.linear_model, ndarray, ndarray)
            first variable is the learned classifier,
            second is a vector of length |num_prev_imgs| *
            len(|prev_context|) + len(|cur_context|)
            third is a vector of length at MOST len(|ordered_dataset[0].ravel|)
    '''
    prev_context_indices = name_to_context_pixels(prev_context)
    current_context_indices = name_to_context_pixels(cur_context)
    assert predictor_family in ['linear', 'logistic'], "Only linear and logistic predictors are currently supported"

    training_context = None
    true_pixels = None
    if not should_extract_training_pairs:
        assert training_filenames is not None, "Must pass filenames for training features and labels if not extracting again"
        training_context = np.load(training_filenames[0], allow_pickle=True, mmap_mode='r')
        true_pixels = np.load(training_filenames[1], allow_pickle=True, mmap_mode='r')
    else:
        start = timer()
        training_context, true_pixels = extract_training_pairs(ordered_dataset,
            num_prev_imgs, prev_context_indices, current_context_indices)
        training_context, true_pixels = np.array(training_context), np.array(true_pixels)
        training_context = training_context.reshape(training_context.shape[0], -1)
        end_extraction = timer()
        print(f'\tExtracted training pairs in ' + \
            f'{timedelta(seconds=end_extraction-start)}.')
        date_str = f'{datetime.now().hour}_{datetime.now().minute}'
        np.save(f'training_context_{date_str}', np.array(training_context))
        np.save(f'true_pixels_{date_str}', np.array(true_pixels))
    start = timer()
    clf = None
    if predictor_family == 'linear':
        clf = linear_model.LinearRegression()
    elif predictor_family == 'logistic':
        # training_context = csr_matrix(training_context)
        clf = linear_model.SGDClassifier(loss='log', n_jobs=-1)
    clf.fit(training_context, true_pixels)
    end_model_fitting = timer()
    print(f'\tTrained a {predictor_family} model in ' + \
        f'{timedelta(seconds=end_model_fitting-start)}.\n')
    # print('\t\t(Accuracy: %05f)\n' %  __compute_classifier_accuracy(clf, predictor_family, training_context, true_pixels))
    return ordered_dataset, 0, (clf, training_context, true_pixels,
        num_prev_imgs, prev_context, cur_context)
