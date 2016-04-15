#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    plots
    ~~~~~

    plotting code

    :license: BSD
"""

__version__ = '0.0.1'

import matplotlib.pyplot as plt

SIDE = 96

def show_img(X, Y, index):
    """show an image instance. """

    imgplot = plt.imshow(X.iloc[index].reshape(SIDE, SIDE), cmap=plt.cm.gray)
    plt.plot((Y.iloc[index], Y.iloc[index]), (0, SIDE))
