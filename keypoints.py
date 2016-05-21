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
import data
import pipeline


def run(config):
    """run the pipeline. """

    p = pipeline.Pipeline(
        config['train'],
        config['test'],
        config['lookup'],
        config['predictions']
    )

    p.run()


def cfg():
    """load the configuration"""

    parser = argparse.ArgumentParser(description='learn and predict.')
    parser.add_argument('--train',       default=data.TRAIN_FILE)
    parser.add_argument('--test',        default=data.TEST_FILE)
    parser.add_argument('--predictions', default=pipeline.PRED_FILE)
    parser.add_argument('--lookup',      default=data.LOOKUP_FILE)

    return vars(parser.parse_args())


if __name__ == '__main__':
    config = cfg()
    run(config)
