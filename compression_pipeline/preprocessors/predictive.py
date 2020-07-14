'''
preprocessors/predictive.py

This module contains the Predictive Coding preprocessor.
'''
import numpy as np
from sklearn import linear_model

from datetime import timedelta, datetime
from timeit import default_timer as timer

import pickle

def name_to_context_pixels(name):
    if name == 'DAB':
        return [(-1, 0), (-1, -1), (0, -1)]
    if name == 'DABC':
        return [(-1, 0), (-1, -1), (0, -1), (1, -1)]
    return None

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


def get_valid_pixels(img_shape, relative_indices):
    '''
    TODO: if raster scans are more general.

    Because we do not currently support scan patterns or initial context that is
    "ahead of"  (either to the right or below) a current pixel, we require
    relative context indices to be negative-valued in the row or zero in the current row 
    but negative in the column.
    '''
    err_msg = "Impossible to satisfy passing initial context with these relative indices"
    assert np.all([index[1] < 0 or (index[1] == 0 and index[0] < 0) for index in relative_indices]), err_msg
    valid_pixels = []
    min_x = abs(min([index[0] for index in relative_indices]))
    max_x = img_shape[1] - max(0, max([index[0] for index in relative_indices])) - 1
    min_y = abs(min([index[1] for index in relative_indices]))
    max_y = img_shape[0] - max(0, max([index[1] for index in relative_indices])) - 1
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            valid_pixels += [(x, y)]
    return valid_pixels

def extract_training_pairs(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
    '''
    Assumes |prev_context_indices| and |current_context_indices| are lists of relative indices on (i, j)
    '''
    valid_in_cur = get_valid_pixels(ordered_dataset[0].shape, current_context_indices)
    valid_in_prev = get_valid_pixels(ordered_dataset[0].shape, prev_context_indices)
    pixels_to_predict = [p for p in valid_in_prev if p in valid_in_cur]
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

def train_linear_reg_predictor(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
    '''
    Linear regression preprocessor

    Args:
        ordered_dataset: numpy array
            data to be preprocessed, of shape (n_elements, n_points)
        num_prev_imgs: int
                how many images preceeding each element should be considered as "dataset context"
        prev_context_indices: list of tuples where each tuple has dimension ordered_dataset.shape[1:]
                each tuple describes the relative location of a pixel to be used for context in each
                "dataset context" image
                current_context_indices: list of tuples where each tuple has dimension ordered_dataset.shape[1:]
                each tuple describes the relative location of a pixel to be used for context
                in the current image

    Returns:
        ordered_dataset: numpy array
            unchanged from input
        element_axis: int
            index into ordered_dataset.shape for n_elements
        (clf, training_context, true_pixels): tuple of (sklearn.linear_model.Lasso, ndarray, ndarray)
            first variable is the learned classifier and
            second is a vector of length |num_prev_imgs| * len(|prev_context_indices|) + len(|current_context_indices|)
            third is a vector of length at MOST len(|ordered_dataset[0].ravel|)
    '''
    start = timer()
    training_context, true_pixels = extract_training_pairs(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices)
    end_extraction = timer()
    print(f'\tExtracted training pairs in {timedelta(seconds=end_extraction-start)}')
    np.save(f'trainingpairs_{datetime.now().hour}_{datetime.now().minute}', (training_context, true_pixels))
    start = timer()
    clf = linear_model.LinearRegression()
    clf.fit(training_context, true_pixels)
    end_model_fitting = timer()
    print(f'\tTrained a linear model in {timedelta(seconds=end_model_fitting-start)}')
    print('Accuracy: %05f' % clf.score(training_context, true_pixels))
    try:
        with open(f'clf_{datetime.now().hour}_{datetime.now().minute}.pickle', 'wb') as f:
            pickle.dump(clf, f)
    except:
        print('\tCouldn\'t pickle clf.')
    return ordered_dataset, 0, (clf, training_context, true_pixels)

