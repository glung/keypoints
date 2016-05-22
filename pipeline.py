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
import json

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import cross_validation

import features
import data
import submit

PRED_FILE = 'predictions.csv'
EVAL_FILE = 'evaluation.txt'


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
        self.seed = 0

        if self.key is None:
            self.key = str(time.time())

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if not os.path.exists(self.build_dir()):
            os.makedirs(self.build_dir())


    def run(self):
        """run the pipeline. """

        # data
        Xtrain, Ytrain, header = data.df_train(self.train_file)

        # evaluate model
        self.evaluate(model, Xtrain, Ytrain)

        # train, predict
        estimator = self.train(self.model(), Xtrain, Ytrain)
        Ypred = self.predict(estimator, data.df_predict(self.test_file))

        # log results
        self.log_evaluation()
        self.submit(header, Ypred)


    def model(self):
        """construct the model. """

        return linear_model.LinearRegression()


    def evaluate(self, model, X, Y):
       folds = cross_validation.KFold(
         len(X),
         n_folds = 10, # 10 is optimal
         random_state = self.seed
       )

       scores = cross_validation.cross_val_score(
           model,
           X,
           Y,
           cv = folds,
           scoring = 'mean_squared_error')

       rmses = np.sqrt(np.abs(scores))
       std = np.std(rmses)
       mean = rmses.mean()

       self.evaluation = {
           'values': ",".join(map(str, rmses)),
           'mean': mean,
           'std': std,
           'upper_confidence_bound': mean + std,
           'lower_confidence_bound': mean - std
       }


    def train(self, model, X, Y):
        """train a model. """

        model.fit(X, Y)

        return model


    def predict(self, model, X):
        """predict. """

        predictions = model.predict(X)

        return pd.DataFrame(features.crop(predictions), index = X.index)


    def submit(self, header, Ypred):
        """generate submission file. """

        lookup = data.df_lookup(header, self.lookup_file)
        S = submit.submission(Ypred, lookup)
        S.to_csv(self.path(PRED_FILE), index = False)


    def log_evaluation(self):
        """results evaluation file. """

        with open(self.path(EVAL_FILE), 'a') as f:
            f.write(str(self.evaluation))


    def path(self, filename):
        return '%s/%s' % (self.build_dir(), filename)


    def build_dir(self):
        return '%s/%s' % (self.results_dir, self.key)
