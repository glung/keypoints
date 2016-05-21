# -*- coding: utf-8 -*-

"""
    pipeline
    ~~~~~~~~~

    learn, evaluate, predict

    :license: BSD
"""

__version__ = '0.0.1'

import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn import linear_model

import features
import data

PRED_FILE = "data/predictions.csv"


class Pipeline():
    """
    Learning pipeline for the kepoints challenge.
    """

    def __init__(
        self,
        train,
        test,
        lookup,
        predictions):

        self.train = train
        self.test = test
        self.lookup = lookup
        self.predictions = predictions


    def run(self):
        """run the pipeline. """

        # learn, predict
        Xtrain, Ytrain, header = data.df_train(self.train)
        model = train(Xtrain, Ytrain)
        Ypred = predict(model, data.df_test(self.test))

        # submit
        lookup = data.df_lookup(header, self.lookup)
        S = submission_format(Ypred, lookup)
        S.to_csv(self.predictions, index = False)


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

    return pd.DataFrame(crop(model.predict(X)), index = X.index)


def crop(df):
    """crop to min and max"""

    df[df > 96] = 96
    df[df < 0] = 0

    return df

