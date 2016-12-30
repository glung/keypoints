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

from sklearn import cross_validation
from sklearn import pipeline as skp

from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.preprocessing import FunctionTransformer

import data
import features
import submit
import model_linear_regression_sklearn

RESULTS_DIR = 'results'
PRED_FILE = 'predictions.csv'
EVAL_FILE = 'evaluation.txt'

CV_FOLDS = 10 # optimal
TRAINING_FRACTION = 1.0 - 1.0 / CV_FOLDS


class Pipeline():
    """
    Learning pipeline for the kepoints challenge.
    """

    def __init__(
        self,
        train_file = data.TRAIN_FILE,
        test_file = data.TEST_FILE,
        lookup_file = data.LOOKUP_FILE,
        results_dir = RESULTS_DIR,
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
        self.evaluate(self.model(), Xtrain, Ytrain)

        # train, predict
        estimator = self.train(self.model(), Xtrain, Ytrain)
        Ypred = self.predict(estimator, data.df_predict(self.test_file))

        # log results
        self.log_evaluation()
        self.submit(header, Ypred)


    def model(self):
        return model_linear_regression_sklearn.linearRigeRegression()


    def preprocessing(self):
        """construct the preprocessing pipeline. """

        return skp.Pipeline([
            ('scaling', FunctionTransformer(features.preprocessing))
        ])


    def validation_curve(self, sample = None):
        """generate validation curve. """

        # read
        Xtrain, Ytrain, header = data.df_train(self.train_file, sample)

        # validation_curve
        param_range = np.logspace(-2, 8, 30)

        # train
        train_scores, test_scores = validation_curve(
            self.model(),
            Xtrain,
            Ytrain,
            param_name = 'regressor__alpha',
            param_range = param_range,
            cv = CV_FOLDS,
            scoring = 'mean_squared_error',
            n_jobs = 1
        )

        return (
            param_range,
            np.sqrt(np.abs(train_scores)),
            np.sqrt(np.abs(test_scores)),
        )


    def learning_curves(self, iterations = 10, sample = None):
        """return learning curves. """

        # read
        Xtrain, Ytrain, header = data.df_train(self.train_file, sample)

        # account for cv folds
        upper = int(len(Xtrain) * TRAINING_FRACTION)

        # learning curve
        train_sizes = np.linspace(
            1,
            upper,
            num = iterations + 1,
            dtype = np.int)[1:]

        # train
        train_sizes, train_scores, test_scores = learning_curve(
            self.model(),
            Xtrain,
            Ytrain,
            train_sizes = train_sizes,
            cv = CV_FOLDS,
            scoring = 'mean_squared_error'
        )

        return (
            train_sizes,
            np.sqrt(np.abs(train_scores)),
            np.sqrt(np.abs(test_scores)),
        )


    def evaluate(self, model, X, Y):
       folds = cross_validation.KFold(
         len(X),
         n_folds = CV_FOLDS, # 10 is optimal
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
           'values': rmses,
           'mean': mean,
           'std': std,
           'upper_confidence_bound': mean + std,
           'lower_confidence_bound': mean - std
       }


    def train(self, model, X, Y):
        """train a model. """

        print (X.shape)
        model.fit(X, Y)

        return model


    def predict(self, model, X):
        """predict. """

        predictions = model.predict(X)

        return pd.DataFrame(features.crop(predictions))


    def submit(self, header, Ypred):
        """generate submission file. """

        lookup = data.df_lookup(header, self.lookup_file)
        S = submit.submission(Ypred, lookup)
        S.to_csv(self.path(PRED_FILE), index = False)


    def log_evaluation(self):
        """results evaluation file. """

        with open(self.path(EVAL_FILE), 'a') as f:
            f.write(str(self.pp_evaluation()))


    def pp_evaluation(self, fmt = '{0:.3f}', indent = 2):
        """pretty print evaluation. """

        values = self.evaluation['values']
        mean = self.evaluation['mean']
        std = self.evaluation['std']
        upper_confidence_bound = self.evaluation['upper_confidence_bound']
        lower_confidence_bound = self.evaluation['lower_confidence_bound']

        formatted = {
           'values': ",".join(map(lambda x: fmt.format(x), values)),
           'mean': fmt.format(mean),
           'std': fmt.format(std),
           'upper_confidence_bound': fmt.format(upper_confidence_bound),
           'lower_confidence_bound': fmt.format(lower_confidence_bound)
        }

        return json.dumps(formatted, indent = indent)


    def path(self, filename):
        return '%s/%s' % (self.build_dir(), filename)


    def build_dir(self):
        return '%s/%s' % (self.results_dir, self.key)
