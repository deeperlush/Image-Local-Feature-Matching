import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import plot_matches


def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, good_matches, number_to_display, filename=None):
	"""
		Visualizes corresponding point