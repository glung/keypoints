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
N_SIDE = 4

def show_imgs(X, Y):
    fig = plt.figure(figsize=(16, 16))

    for i in range(N_SIDE * N_SIDE):
        ax = fig.add_subplot(N_SIDE, N_SIDE, i + 1, xticks=[], yticks=[])
        show_img(X.iloc[i], Y.iloc[i], ax)


def show_img(x, y, axis):
    """show an image instance. """

    img = x.reshape(SIDE, SIDE)
    axis.imshow(img, cmap=plt.cm.gray)
    axis.scatter(y[0::2], y[1::2], marker = '.', s = 10, color = 'red')
