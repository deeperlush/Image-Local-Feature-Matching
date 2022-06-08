import memory_profiler
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from skimage.transform import rescale

import student
from helpers import evaluate_correspondence


def memfunc():
    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_