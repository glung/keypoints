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


def show_preprocessed_imgs(p, X, Y, color = 'red'):
    Xpreprocessed = pd.DataFrame(
        p.preprocessing().fit_transform(X),
        index = X.index
    )

    show_imgs(Xpreprocessed, Y, color)


def show_imgs(X, Y, color = 'red', nb=N_SIDE * N_SIDE):
    """show the first N_SIDE ^ 2 images in X. """

    show_imgs_array(X.iloc, Y.iloc, len(X), color, nb)


def show_imgs_array(X, Y, len, color = 'red', nb=N_SIDE * N_SIDE):
    """show the first N_SIDE ^ 2 images in X. """

    fig = plt.figure(figsize=(16, 16))
    for i in range(min(nb, len)):
        ax = fig.add_subplot(N_SIDE, N_SIDE, i + 1, xticks=[], yticks=[])
        show_img(X[i], Y[i], ax, color)


def show_img(x, y, axis, color):
    """show an image instance. """

    img = x.reshape(SIDE, SIDE)
    axis.imshow(img, cmap=plt.cm.gray)
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


def plot_learning_curves(train_sizes, train_scores, test_scores):
    """plot learning curves data as returned by the pipeline"""

    plt.figure(figsize=(16, 8))
    plt.title('Learning curves')
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # summaries
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plotit
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha = 0.1,
        color = 'r')

    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha = 0.1,
        color = 'g')

    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color = 'r',
        label = 'Training score')

    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color = 'g',
        label = 'Cross-validation score')

    plt.legend(loc = 'best')

    return plt


def plot_validation_curve(param_range, train_scores, test_scores):
    """plot validation curves with respect to Ridge alpha"""

    plt.figure(figsize=(16, 8))
    plt.title('Validation Curve')
    plt.xlabel('alpha')
    plt.ylabel('Score')

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)

    plt.semilogx(
        param_range,
        train_scores_mean,
        label = 'Training score',
        color = 'r'
    )

    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha = 0.2,
        color = 'r'
    )

    plt.semilogx(
        param_range,
        test_scores_mean,
        label = 'Cross-validation score',
        color = 'g'
    )

    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha = 0.2,
        color = 'g'
    )

    plt.legend(loc = 'best')

    plt.show()
