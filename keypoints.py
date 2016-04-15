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

    model = linear_model.LogisticRegression()
    model.fit(X, Y)

    return model


def predict(model, ids, X):
    """predict. """

    Y = model.predict(X)

    return pd.DataFrame(data = { 'ID': ids, 'TARGET': Y })


def df_train(filename = TRAIN_FILE):
    """the train set. """

    df = pd.read_csv(filename, header = 0)
    X = df.drop(['ID', 'TARGET'], axis = 1)
    Y = df['TARGET']

    return (X, Y)


def df_test(filename = TEST_FILE):
    """the test set. """

    df = pd.read_csv(filename, header = 0)
    ids = df['ID']
    X = df.drop(['ID'], axis = 1)

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
