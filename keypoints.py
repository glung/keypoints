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
LOOKUP_FILE = "data/IdLookupTable.csv"


def run(config):
    """run prediction. """

    # learn, predict
    Xtrain, Ytrain, header = df_train(config['train'])
    model = train(Xtrain, Ytrain)
    Ypred = predict(model, df_test(config['test']))

    # submit
    lookup = df_lookup(header, config['lookup'])
    S = submission_format(Ypred, lookup)
    S.to_csv(config['predictions'], index = False)


def submission_format(Y, lookup):
    """
    convert to the right format for submission. joins id lookup table to
    the predictions like this:

    prediction: (imageid, featureid, location) ­ unstacks first.
    lookup: (rowid, imageid, featureid)
    joined: (rowid, location)
    """

    joined = lookup.join(unstacked_predictions(Y)).reset_index()

    predictions = OrderedDict([
        ('RowId',    joined['RowId']),
        ('Location', joined['Location'])
    ])

    return pd.DataFrame(data = predictions)


def unstacked_predictions(Y):
    """
    unstack predictions to prepare for join with lookup table.
    """

    keys = pd.DataFrame(Y.transpose().unstack())
    keys.reset_index(inplace = True)
    keys.columns = ['ImageId', 'FeatureId', 'Location']

    return keys.set_index(['ImageId', 'FeatureId'])


def train(X, Y):
    """train a model. """

    model = linear_model.LinearRegression()
    model.fit(X, Y)

    return model


def predict(model, X):
    """predict. """

    return pd.DataFrame(model.predict(X))


def to_image(str_images):
    """convert string representation to vector"""

    return pd.Series(map(lambda v: float(v), str_images.split()))


def df_train(filename = TRAIN_FILE, sample = None):
    """the train set. """

    df = pd.read_csv(filename, header = 0)
    if sample is not None:
        df = df.sample(sample)

    df = df.dropna()
    Y = df.drop('Image', axis = 1)
    X = df['Image'].apply(to_image)
    header = Y.columns.values


    return (X, Y, header)


def df_test(filename = TEST_FILE):
    """the test set. """

    df = pd.read_csv(filename, header = 0, index_col = 'ImageId')
    return df['Image'].apply(to_image)


def df_lookup(header, filename = LOOKUP_FILE):
    """read mapping file and covert to (row id, image id, feature id). """

    lookup = dict(zip(header, range(0,30)))
    df = pd.read_csv(filename, header = 0, index_col = 'RowId')
    df['FeatureId'] = df.FeatureName.apply(lambda name: lookup[name] )
    df = df.drop(['FeatureName', 'Location'], axis = 1).reset_index()

    return df.set_index(['ImageId', 'FeatureId'])


def cfg():
    """load the configuration"""

    parser = argparse.ArgumentParser(description='learn and predict.')
    parser.add_argument('--train',       default=TRAIN_FILE)
    parser.add_argument('--test',        default=TEST_FILE)
    parser.add_argument('--predictions', default=PRED_FILE)
    parser.add_argument('--lookup',      default=LOOKUP_FILE)

    return vars(parser.parse_args())


if __name__ == '__main__':
    run(cfg())
