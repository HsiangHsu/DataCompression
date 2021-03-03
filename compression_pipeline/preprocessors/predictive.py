'''
preprocessors/predictive.py

This module contains the Predictive Coding preprocessor.
'''
import numpy as np
from sklearn import linear_model
from scipy.sparse import csr_matrix

from datetime import timedelta, datetime
from timeit import default_timer as timer

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

import pickle
import re

from utilities import name_to_context_pixels, predictions_to_pixels, \
    get_valid_pixels_for_predictions

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
        remaining_samples_to_predict = len(true_pixels)
        right_pixels_count = 0
        start_index = 0
        while remaining_samples_to_predict > 0:
            predict_batch_size = min(remaining_samples_to_predict, 1000)
            dtype = training_context.dtype
            estimated_pixels = predictions_to_pixels(clf.predict(training_context[start_index:start_index + predict_batch_size]), dtype)
            estimated_pixels = estimated_pixels.reshape(true_pixels[start_index:start_index + predict_batch_size].shape)
            if len(estimated_pixels.shape) > 1:
                right_pixels_count += np.count_nonzero(np.all(estimated_pixels == true_pixels[start_index:start_index + predict_batch_size], axis=1))
            else:
                right_pixels_count += np.count_nonzero(estimated_pixels == true_pixels[start_index:start_index + predict_batch_size])
            start_index += predict_batch_size
            remaining_samples_to_predict -= predict_batch_size
        return right_pixels_count / len(true_pixels)
    assert False, 'Must be a logistic or linear predictor to compute accuracy'
    

def train_predictor(predictor_family, ordered_dataset, num_prev_imgs,
                    prev_context, cur_context, mode, num_cubist_rules,
                    should_extract_training_pairs=True,
                    training_filenames=None,
                    should_train=True, predictor_filename=False):
    '''
    Generalized predictive coding preprocessor

    Args:
        predictor_family: str
            one of 'linear', 'logistic', 'cubist' for the regression family 
            to compute across training features and labels
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
        mode: string
            strategy for setting up predictors, in particular for RGB images
        num_cubist_rules: int or None
            for |predictor_family| 'cubist', the maximum number of linear 
            predictors to subdivide
        should_extract_training_pairs: boolean (default True)
            whether to compute the training context from the dataset or else
            load from |training_pairs_filename|
        training_filenames: tuple of (string, string) (default None)
            if |should_extract_training_pairs|, the locations of the NumPy
            files from which to load |training_context| and |true_pixels|,
            respectively
        should_train: boolean (default True)
            whether to train the predictor from the training context or
            load from |predictor_filename|
        predictor_filename: string (default None)
            if |should_train|, the locations of the Pickle dumps
            from which to load |clf|

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
    assert predictor_family in ['linear', 'logistic', 'cubist', 'quantile'], \
        "Only linear, logistic, and cubist predictors are currently supported"
    assert num_cubist_rules is None if predictor_family != 'cubist' else True, \
        "Can only specify |num_cubist_rules| when using the cubist predictor"
    if mode == 'triple':
        assert ordered_dataset.ndim == 4 and ordered_dataset.shape[-1] == 3, \
            'Invalid data shape for triple mode.'
        n_pred = 3
    elif mode == 'single':
        n_pred = 1
    if predictor_family == 'cubist':
        n_pred *= num_cubist_rules

    date_str = f'{datetime.now().hour}_{datetime.now().minute}'

    start = timer()
    if not should_extract_training_pairs:
        assert training_filenames is not None, \
            "Must pass filenames for training features and labels if not " + \
            "extracting again"
        training_context = np.load(training_filenames[0], allow_pickle=True,
            mmap_mode='r')
        true_pixels = np.load(training_filenames[1], allow_pickle=True,
            mmap_mode='r')
        end_extraction = timer()
        print(f'\tLoaded training pairs in ' + \
            f'{timedelta(seconds=end_extraction-start)}.')
        # TODO: Validate loaded training context and true pixels to ensure they
        #       match shape, etc. of parameters and data passed in
    else:
        training_context, true_pixels = extract_training_pairs(ordered_dataset,
            num_prev_imgs, prev_context_indices, current_context_indices)
        training_context, true_pixels = np.array(training_context), \
            np.array(true_pixels)
        training_context = \
            training_context.reshape(training_context.shape[0], -1)
        if mode == 'triple':
            true_pixels = true_pixels.transpose((1,0))
        elif mode == 'single':
            true_pixels = true_pixels.reshape((1, *true_pixels.shape))
        end_extraction = timer()
        print(f'\tExtracted training pairs in ' + \
            f'{timedelta(seconds=end_extraction-start)}.')

        np.save(f'training_context_{date_str}', np.array(training_context))
        np.save(f'true_pixels_{date_str}', np.array(true_pixels))

    start = timer()
    # TODO(cubist): can we save/load?
    if not should_train:
        assert predictor_filename is not None, \
            'Must pass filenames for predictor if not training again.'
        with open(predictor_filename, 'rb') as f:
            clf = pickle.load(f)
        predictors = {'LinearRegression': 'linear',
            'SGDClassifier': 'logistic'}
        pred_name = str(clf[0]).split('(')[0]
        predictor_family = predictors[pred_name]
        end_model_loading = timer()
        print(f'\tLoaded a {predictor_family} model in ' + \
            f'{timedelta(seconds=end_model_loading-start)}.')
    else:
        if predictor_family == 'linear':
            clf = [linear_model.LinearRegression(n_jobs=-1) \
                for i in range(n_pred)]
            for i in range(n_pred):
                clf[i].fit(training_context, true_pixels[i])
        elif predictor_family == 'logistic':
            training_context = csr_matrix(training_context)
            clf = [linear_model.SGDClassifier(loss='log', n_jobs=-1) \
                for i in range(n_pred)]
            for i in range(n_pred):
                clf[i].fit(training_context, true_pixels[i])
        elif predictor_family == 'cubist':
            clf = []
            # Required R packages
            packnames = ('Cubist', 'tidyrules')
            utils = importr('utils')
            # Selectively install what needs to be install.
            names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
            if len(names_to_install) > 0:
                    # select a mirror for R packages (first in list)
                    utils.chooseCRANmirror(ind=1) 
                    utils.install_packages(StrVector(names_to_install))

            rpy2.robjects.numpy2ri.activate()
            Cubist = importr('Cubist')
            tidyRules = importr('tidyrules')
            r = robjects.r
            # Set maximum number of linear models we want
            cc = r['cubistControl'](rules=5)

            nr,num_training_features = training_context.shape
            img_labels = ["(image %d)" % i for i in range(nr)]
            feature_labels = ["(context %d)" % i for i in range(num_training_features)]
            Xr = r.matrix(training_context, nrow=nr, ncol=num_training_features,
                                   dimnames=[img_labels, feature_labels])
            r.assign("training_context", Xr)
            
            nr,nc = true_pixels.shape
            Yr = robjects.r.matrix(true_pixels, nrow=nr, ncol=nc,
                                   dimnames=[["pixel value"], img_labels])
            r.assign("true_pixels", Yr)
            regr = r['cubist'](x=Xr, y=Yr, control=cc)

            models = r['tidyRules'](regr)
            assert len(models['LHS']) == n_pred, \
                    "Cubist model resulted in a different number of rules than specified"
            for i in range(n_pred):
                    predicate = models['LHS'][i].split("&")
                    for j in range(len(predicate)):
                        predicate[j] =  re.split("\(`\(context\s+", predicate[j])
                        predicate[j] = ''.join([p for p in predicate[j] if len(p) > 0])
                        feature_ind = int(re.findall('[\d]+', predicate[j])[0])
                        tmp = re.split('`\)', predicate[j])
                        tmp = ''.join(tmp[1:])
                        predicate[j] = "x[%i] %s" % (feature_ind, tmp)
                    predicate = ' and '.join(predicate)
                    
                    model_str = models['RHS'][i].replace("- (", "+ (-")
                    parsed_model = [c.split("*") for c in re.split("[+]\s", model_str)]
                    intercept = float(parsed_model[0][0].replace(")", "").replace("(", "").strip())
                    coefs = np.full((num_training_features,), 0.0)
                    parsed_model = parsed_model[1:]
                    for c in parsed_model:
                            coefs[int(re.findall("\d+", c[1])[0])] = float(c[0].replace(")", "").replace("(", "").strip())
                    lin_model = linear_model.LinearRegression()
                    lin_model.intercept_ = np.float64(intercept)
                    lin_model.coef_ = np.array(coefs)
                    clf.append((predicate, lin_model))
            rpy2.robjects.numpy2ri.deactivate()
        elif predictor_family == "quantile":
            clf = []
            q = np.quantile(true_pixels, np.linspace(0,1,31),interpolation='nearest')
            print("Quantiles: "+str(q))

            # map to quantile
            Yq = np.argmin(np.abs((true_pixels.reshape(-1,1).astype(np.int16)-q.reshape(1,-1).astype(np.int16))),axis=1)

            # number of quantiles
            quantiles, counts = np.unique(Yq, return_counts=True) 
            num_quantiles = len(quantiles)

            print("Distribution: "+str(counts/len(Yq)))
            print("Quantiles: "+str(list(set(q))))

            # one hot encode
            enc = OneHotEncoder()
            Yc = enc.fit_transform(Yq.reshape(-1,1)).todense()

            # Create Keras model
            model = Sequential()
            model.add(Dense(256, activation='relu',input_dim=training_context.shape[1], use_bias=True,bias_initializer="zeros"))
            model.add(Dense(256, activation='relu', use_bias=True,bias_initializer="zeros"))
            model.add(Dense(num_quantiles,activation="softmax",use_bias=True,bias_initializer="zeros"))
            loss_fn = keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss_fn, optimizer='adam', metrics=["accuracy"])
            history = model.fit(training_context, Yc, epochs=10, batch_size=50000,verbose=2)

            # save weights
            weights = model.get_weights()
            file = 'keras-weights-mnist-quantile.npz'
            np.savez_compressed(file, weights=weights)
            clf = [model]
        else:
            print("ERROR: unknown predictor_family %s" % predictor_family)
            quit()

        end_model_fitting = timer()
        print(f'\tTrained a {predictor_family} model in ' + \
            f'{timedelta(seconds=end_model_fitting-start)}.')

        if predictor_family == "quantile":
                print('\t\t(Accuracy (in last epoch): %05f)' % history.history['accuracy'][-1])
                '''
                guesses = np.argmax(clf[0].predict(training_context, batch_size=50000), axis=1)
                print("a single guess before argmax looks like", clf[0].predict(training_context[0]))
                correct_pixels = np.count_nonzero(np.ravel(guesses.astype(np.int16))-np.ravel(true_pixels[0].astype(np.int16)))
                print("true_pixels[0].shape", true_pixels[0].shape)
                print('\t\t(Accuracy : %05f)' % correct_pixels/true_pixels[0].shape[0]) '''
        if predictor_family in ['linear', 'logistic']:
                with open(f'predictor_{date_str}.out', 'wb') as f:
                        pickle.dump(clf, f)

                for i in range(n_pred):
                        print('\t\t(Accuracy: %05f)' % \
                              __compute_classifier_accuracy(clf[i],
                                                            predictor_family, training_context, true_pixels[i]))
    print()

    return ordered_dataset, 0, (clf, training_context, true_pixels,
        num_prev_imgs, prev_context, cur_context)
