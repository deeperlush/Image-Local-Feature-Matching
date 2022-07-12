
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

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    
    a = 0.06
    threshold = 0.005
    stride = 2
    sigma = 0.2
    rows = image.shape[0]
    cols = image.shape[1]
    xs = []
    ys = []
    print("R threshold: ", threshold, " Stride: ", stride, "Gaussian sigma: ", sigma, " Alpha: ", a)
    #get the gradients in the x and y directions using sobel filter
    # image = gaussian(image)
    USE_SOBEL = True 
    if USE_SOBEL:
        I_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
        I_y = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)
    else:
        I_x = np.abs(cv2.Scharr(image, cv2.CV_32F, 1, 0))
        I_y = np.abs(cv2.Scharr(image, cv2.CV_32F, 0, 1))
    I_x = gaussian(I_x, sigma)
    I_y = gaussian(I_y, sigma)

    Ixx = I_x**2
    Ixy = I_y*I_x
    Iyy = I_y**2

    # find the sum squared difference (SSD)
    for y in range(0,rows-feature_width,stride):
        for x in range(0,cols-feature_width,stride):
            Sxx = np.sum(Ixx[y:y+feature_width+1, x:x+feature_width+1])
            Syy = np.sum(Iyy[y:y+feature_width+1, x:x+feature_width+1])
            Sxy = np.sum(Ixy[y:y+feature_width+1, x:x+feature_width+1])
            #Find determinant and trace, use to get corner response
            detH = (Sxx * Syy) - (Sxy**2)
            traceH = Sxx + Syy
            R = detH - a*(traceH**2)
            #If corner response is over threshold, it is a corner
            if R > threshold:
                xs.append(x + int(feature_width/2 -1))
                ys.append(y + int(feature_width/2 -1))
    return np.asarray(xs), np.asarray(ys)


def get_features(image, xs, ys, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    # Convert to integers for indexing 
    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)

    # Define helper functions for readabilty and avoid copy-pasting
    def get_window(y, x):
        """
         Helper to get indices of the feature_width square
        """
        rows = (x - (feature_width/2 -1), x + feature_width/2)
        if rows[0] < 0:
            rows = (0, rows[1] - rows[0])
        if rows[1] >= image.shape[0]:
            rows = (rows[0]  + (image.shape[0] -1 - rows[1]), image.shape[0] - 1)
        cols = (y - (feature_width/2 -1), y + feature_width/2)
        if cols[0] < 0:
            cols = (0, cols[1] - cols[0])
        if cols[1] >= image.shape[1]:
            cols = (cols[0]  - (cols[1] + 1 - image.shape[1]), image.shape[1] - 1)
        return int(rows[0]), int(rows[1]) + 1, int(cols[0]), int(cols[1]) + 1

    def get_current_window(i, j, matrix):
        """
        Helper to get sub square of size feature_width/4 
        From the square matrix of size feature_width
        """
        return matrix[int(i*feature_width/4):
                    int((i+1)*feature_width/4),
                    int(j*feature_width/4):
                    int((j+1)*feature_width/4)]

    def rotate_by_dominant_angle(angles, grads):
        hist, bin_edges = np.histogram(angles, bins= 36, range=(0, 2*np.pi), weights=grads)
        angles -= bin_edges[np.argmax(hist)]
        angles[ angles < 0 ] += 2 * np.pi
    
    # Initialize features tensor, with an easily indexable shape
    features = np.zeros((len(xs), 4, 4, 8))
    # Get gradients and angles by filters (approximation)
    sigma = 0.8
    filtered_image = gaussian(image, sigma)
    dx = scharr_v(filtered_image)