# -*- coding: utf-8 -*-

"""

    features
    ~~~~~~~~~

    feature engineering

    :license: BSD
"""

__version__ = '0.0.1'

import numpy as np

from skimage import exposure, filters
from skimage.filters import sobel, threshold_otsu
from skimage.exposure import rescale_intensity

SIDE = 96


def preprocessing(X):
    """apply image normalize to a 2d numpy array"""

    normalized = np.apply_along_axis(image_normalize, 1, X)
    binarized = np.apply_along_axis(binarize, 1, normalized)

    return binarized


def binarize(X):
    """binarize the image black/white after edge detection"""

    img = rescale_intensity(1 - sobel(X.reshape(SIDE, SIDE)))
    thresh = threshold_otsu(img)
    binary = img > thresh

    return binary.flatten()


def image_normalize(X):
    """equalize histogram. """

    img = X.reshape(SIDE, SIDE)
    return exposure.equalize_hist(img).flatten()


def crop(df):
    """crop to min and max"""

    df[df > SIDE] = SIDE
    df[df < 0] = 0

    return df

