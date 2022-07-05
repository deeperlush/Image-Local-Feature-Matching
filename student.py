
import numpy as np
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
import cv2
# debug:
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())