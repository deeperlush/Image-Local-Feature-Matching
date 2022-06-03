import memory_profiler
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from skimage.transform import rescale

import student
from helpers import evaluate_correspondence


def mem