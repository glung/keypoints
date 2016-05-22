# -*- coding: utf-8 -*-

"""
    pipeline
    ~~~~~~~~~

    learn, evaluate, predict

    :license: BSD
"""

__version__ = '0.0.1'

import os
import time

import numpy as np
import pandas as pd

from sklearn import linear_model

import features
import data
import submit

PRED_FILE = "predictions.csv"


class Pipeline():
    """
    Learning pipeline for the kepoints challenge.
    """

    def __init__(
        self,
        train_file,
        test_file,
        lookup_file,
        results_dir,
        key = None):

        self.train_file = train_file
        self.test_file = test_file
        self.lookup_file = lookup_file
        self.results_dir = results_dir
        self.key = key

        if self.key is None:
            self.key = str(time.time())

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if not os.path.exists(self.build_dir()):
            os.makedirs(self.build_dir())


    def run(self):
        """run the pipeline. """

        Xtrain, Ytrain, header = data.df_train(self.train_file)
        model = self.train(Xtrain, Ytrain)
        Ypred = self.predict(model, data.df_test(self.test_file))
        self.submit(header, Ypred)


    def train(self, X, Y):
        """train a model. """

        model = linear_model.LinearRegression()
        model.fit(X, Y)

        return model


    def predict(self, model, X):
        """predict. """

        return pd.DataFrame(features.crop(model.predict(X)), index = X.index)


    def submit(self, header, Ypred):
        """generate submission file. """

        lookup = data.df_lookup(header, self.lookup_file)
        S = submit.submission(Ypred, lookup)
        S.to_csv(self.path(PRED_FILE), index = False)


    def path(self, filename):
        return '%s/%s' % (self.build_dir(), filename)


    def build_dir(self):
        return '%s/%s' % (self.results_dir, self.key)
