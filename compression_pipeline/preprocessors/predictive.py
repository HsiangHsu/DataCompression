'''
preprocessors/predictive.py

This module contains the Predictive Coding preprocessor.
'''
import numpy as np

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
	pass


def extract_training_pairs(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
	pass


def train_linear_predictor(ordered_dataset, num_prev_imgs, prev_context_indices, current_context_indices):
	pass