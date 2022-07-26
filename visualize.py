import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import plot_matches


def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, good_matches, number_to_display, filename=None):
	"""
		Visualizes corresponding points between two images, either as
		arrows or dots

		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line

		Writes out a png of the visualization if 'filename' is not None.
	"""

	# generates unique figures so students can
	# look at all three at once
	fig, ax = plt.subplots(nrows=1, ncols=1)

	matches = matches[0:number_to_display, :]
	good_matches = good_matches[0:number_to_display]

	kp1 = zip_x_y(Y1, X1)
	kp2 = zip_x_y(Y2, X2)
	matches = matches.astype(int)
	plot_matches(ax, imgA, imgB, kp1, kp2, m