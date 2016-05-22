# -*- coding: utf-8 -*-

"""
    features
    ~~~~~~~~~

    feature engineering

    :license: BSD
"""

__version__ = '0.0.1'

from skimage import exposure, filters

SIDE = 96


def image_normalize(img):
    """equalize histogram. """

    return exposure.equalize_hist(img.reshape(SIDE, SIDE)).flatten()


def crop(df):
    """crop to min and max"""

    df[df > SIDE] = SIDE
    df[df < 0] = 0

    return df

