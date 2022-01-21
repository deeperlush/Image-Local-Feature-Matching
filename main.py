# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import io, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import student as student
from hel