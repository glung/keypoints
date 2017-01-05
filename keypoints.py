# -*- coding: utf-8 -*-

"""
    keypoints
    ~~~~~~~~~

    learn facial keypoints.

    :license: BSD
"""

__version__ = '0.0.1'

import argparse
import data
import pipeline
import model_linear_regression_sklearn
import model_linear_regression_tensorflow
import model_neural_network_tensorflow

SKLEARN_LINEAR_REGRESSION = "SKLEARN_LINEAR_REGRESSION"
SKLEARN_LINEAR_RIGE = "SKLEARN_LINEAR_RIGE"
TENSORFLOW_CONTRIB_LEARN_LINEAR_REGRESSION = "TENSORFLOW_CONTRIB_LEARN_LINEAR_REGRESSION"
TENSORFLOW_LINEAR_REGRESSION = "TENSORFLOW_LINEAR_REGRESSION"
TENSORFLOW_ONE_HIDDEN_LAYER = "TENSORFLOW_ONE_HIDDEN_LAYER"


def main(config):
    """run the pipeline. """

    p = pipeline.Pipeline(
        model(config),
        config['train'],
        config['test'],
        config['lookup'],
        config['results_dir']
    )

    p.run()


def model(config):
    """Return the requested model"""

    model_name = config['model']
    if model_name == SKLEARN_LINEAR_REGRESSION:
        return lambda : model_linear_regression_sklearn.model_linear_regression()
    if model_name == SKLEARN_LINEAR_RIGE:
        return lambda : model_linear_regression_sklearn.model_linear_rige()
    if model_name == TENSORFLOW_CONTRIB_LEARN_LINEAR_REGRESSION:
        return lambda : model_linear_regression_tensorflow.contrib_learn_linear_regression()
    if model_name == TENSORFLOW_LINEAR_REGRESSION:
        return lambda : model_linear_regression_tensorflow.vanilla_linear_regression()
    if model_name == TENSORFLOW_ONE_HIDDEN_LAYER:
        return lambda : model_neural_network_tensorflow.one_hidden_layer()

    raise Exception('Unknown model')


def cfg():
    """load the configuration"""

    parser = argparse.ArgumentParser(description='learn and predict.')
    parser.add_argument('--model', default=data.TRAIN_FILE)
    parser.add_argument('--train', default=data.TRAIN_FILE)
    parser.add_argument('--test', default=data.TEST_FILE)
    parser.add_argument('--lookup', default=data.LOOKUP_FILE)
    parser.add_argument('--results-dir', default=pipeline.RESULTS_DIR)

    return vars(parser.parse_args())


if __name__ == '__main__':
    config = cfg()
    main(config)
