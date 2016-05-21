# -*- coding: utf-8 -*-

"""
    data
    ~~~~

    read csvs into dataframes.

    :license: BSD
"""

__version__ = '0.0.1'

import pandas as pd

TRAIN_FILE = "data/training.csv"
TEST_FILE = "data/test.csv"
LOOKUP_FILE = "data/IdLookupTable.csv"


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


def to_image(str_images):
    """convert string representation to vector"""

    return pd.Series(map(lambda v: float(v), str_images.split()))
