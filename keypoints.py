#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    keypoints
    ~~~~~~~~~

    learn a image facial keypoints.

    :license: BSD
"""

__version__ = '0.0.1'

import argparse
import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn import linear_model

TRAIN_FILE = "data/training.csv"
TEST_FILE = "data/test.csv"
PRED_FILE = "data/predictions.csv"


def run(config):
    """run prediction. """

    model = train(*df_train(config['train']))
    df = predict(model, *df_test(config['test']))
    df.to_csv(config['predictions'], index = False)


def train(X, Y):
    """train a model. """

    model = linear_model.LinearRegression()
    model.fit(X, Y)

    return model


def predict(model, ids, X):
    """predict. """

    Y = model.predict(X)
    predictions = OrderedDict([( 'RowId', ids), ('Location', Y )])
    return pd.DataFrame(data = predictions)


def to_image(str_images):
    """convert string representation to vector"""

    return pd.Series(map(lambda v: float(v), str_images.split()))


def df_train(filename = TRAIN_FILE, sample = None):
    """the train set. """

    df = pd.read_csv(filename, header = 0)
    if sample is not None:
        df = df.sample(sample)

    df = df.dropna()
    Y = df['left_eye_center_x'].apply(lambda v: float(v))
    X = df['Image'].apply(to_image)

    return (X, Y)


def df_test(filename = TEST_FILE):
    """the test set. """

    df = pd.read_csv(filename, header = 0)
    ids = df['ImageId']
    X = df['Image'].apply(to_image)

    return (ids, X)


def cfg():
    """load the configuration"""

    parser = argparse.ArgumentParser(description='learn and predict.')
    parser.add_argument('--train',       default=TRAIN_FILE)
    parser.add_argument('--test',        default=TEST_FILE)
    parser.add_argument('--predictions', default=PRED_FILE)

    return vars(parser.parse_args())


if __name__ == '__main__':
    run(cfg())
