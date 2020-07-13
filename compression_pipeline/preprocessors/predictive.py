'''
preprocessors/predictive.py

This module contains the Predictive Coding preprocessor.
'''
import numpy as np
from sklearn import linear_model

from datetime import timedelta
from timeit import default_timer as timer

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
	pixels_to_predict = get_valid_pixels(ordered_dataset[0].shape, current_context_indices)
	X_train = []
	Y_train = []
	for current_img_index in range(num_prev_imgs, ordered_dataset.shape[0]):
		for (i, j) in pixels_to_predict:
			predictive_string = []
			for a in range(num_prev_imgs):
				predictive_string.append(ordered_dataset[current_img_index - a - 1][apply_relative_indices(prev_context_indices, i, j)])
			predictive_string.append(ordered_dataset[current_img_index][apply_relative_indices(current_context_indices, i, j)])
			X_train.append(np.array(predictive_string).ravel())
			Y_train.append(ordered_dataset[current_img_index][i][j])
	return (X_train, Y_train)

def train_lasso_predictor(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
	'''
    Lasso linear regression preprocessor

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
        clf: sklearn.linear_model.Lasso 
            learned classifier
    '''
	start = timer()
	training_context, true_pixels = extract_training_pairs(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices)
	end_extraction = timer()
	print(f'\tExtracted training pairs in {timedelta(seconds=end_extraction-start)}')
	np.save('trainingpairs', (training_context, true_pixels))

	start = timer()
	clf = linear_model.Lasso(alpha=0.1)
	clf.fit(training_context, true_pixels)
	end_model_fitting = timer()
	print(f'\tTrained a lasso model in {timedelta(seconds=end_model_fitting-start)}')

	return ordered_dataset, 0, clf.sparse_coef_

