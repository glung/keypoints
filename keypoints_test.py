#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    tests
    ~~~~~

    tests for keypoints

    :license: BSD
"""

__version__ = '0.0.1'

from nose.tools import *

import os
import shutil

import pipeline

FX_TRAIN = 'data/fixture.training.csv'
FX_TEST = 'data/fixture.test.csv'
FX_LOOKUP = 'data/fixture.lookup.csv'
RESULTS_DIR = 'results-test'


def test_integration():
    """test the whole pipeline"""

    p = pipeline.Pipeline(
        FX_TRAIN,
        FX_TEST,
        FX_LOOKUP,
        RESULTS_DIR
    )

    p.run()

    assert os.path.isfile(p.path(pipeline.PRED_FILE))


def teardown_module():
    shutil.rmtree(RESULTS_DIR)
