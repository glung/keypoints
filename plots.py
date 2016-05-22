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
import numpy as np
import pandas as pd

SIDE = 96
N_SIDE = 6


# XXX(rk): move to pipeline.
def show_preprocessed_imgs(p, X, Y, color = 'red'):
    Xpreprocessed = pd.DataFrame(
        p.preprocessing().fit_transform(X),
        index = X.index
    )

    show_imgs(Xpreprocessed, Y, color)


def show_imgs(X, Y, color):
    """show the first N_SIDE ^ 2 images in X. """

    fig = plt.figure(figsize=(16, 16))
    for i in range(N_SIDE * N_SIDE):
        ax = fig.add_subplot(N_SIDE, N_SIDE, i + 1, xticks=[], yticks=[])
        show_img(X.iloc[i], Y.iloc[i], ax, color)


def show_img(x, y, axis, color):
    """show an image instance. """

    img = x.reshape(SIDE, SIDE)
    axis.imshow(binary, cmap=plt.cm.gray)
    axis.scatter(
        y[0::2],
        y[1::2],
        marker = '.',
        s = 10,
        color = color)


def show_model(model, header):
    """show model coefficients"""

    fig = plt.figure(figsize=(16, 16))
    coefs = model.coef_

    for i, c in  enumerate(coefs):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        plt.title(header[i])
        ax.imshow(c.reshape(SIDE, SIDE))
