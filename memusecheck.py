import memory_profiler
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from skimage.transform import rescale

import student
from helpers import evaluate_correspondence


def memfunc():
    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"
    eval_file = "../data/NotreDame/NotreDameEval.mat"

    scale_factor = 0.5
    feature_width = 16

    image1 = img_as_float32(rescale(rgb2gray(io.imread(image1_file)), scale_factor))
    image2 = img_as_float32(rescale(rgb2gray(io.imread(image2_file)), scale_factor))

   