# -*- coding: utf-8 -*-

"""
    data
    ~~~~

    read csvs into dataframes.

    :license: BSD
"""

__version__ = '0.0.1'

import pandas as pd
import numpy as np

TRAIN_FILE = "data/training.csv"
TEST_FILE = "data/test.csv"
LOOKUP_FILE = "data/IdLookupTable.csv"


def df_train(filename = TRAIN_FILE, sample = None, expand=True):
    """the train set. """

    df = pd.read_csv(filename, header = 0)
    if sample is not None:
        df = df.sample(n = sample)

    # FIXME(rk): we're throwing away 70% of our data here.
    df = df.dropna()

    Y = df.drop('Image', axis = 1)
    X = df['Image'].apply(to_image)
    header = Y.columns.values

    if expand:
        (X, Y) = df_train_expand(X, Y)

    return (X, Y, header)

def df_train_expand(X, Y):
    def load2d():
        return np.vstack(X.values).reshape(-1, 1, 96, 96)

    def flip_2d_images_horizontally(images2d):
        return images2d[:, :, :, ::-1]

    def images2d_to_dataframe(images2d):
        return pd.DataFrame(data=images2d.flatten().reshape(X.shape), index=X.index, columns=X.columns)

    def flip_labels_horizontally():
        columns_mapping = {
            'left_eye_center_x'         :    'right_eye_center_x',
            'left_eye_center_y'         :   'right_eye_center_y',
            'left_eye_inner_corner_x'   :   'right_eye_inner_corner_x',
            'left_eye_inner_corner_y'   :   'right_eye_inner_corner_y',
            'left_eye_outer_corner_x'   :   'right_eye_outer_corner_x',
            'left_eye_outer_corner_y'   :   'right_eye_outer_corner_y',
            'left_eyebrow_inner_end_x'  :   'right_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y'  :   'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x'  :   'right_eyebrow_outer_end_x',
            'left_eyebrow_outer_end_y'  :   'right_eyebrow_outer_end_y',
            'mouth_left_corner_x'       :   'mouth_right_corner_x',
            'mouth_left_corner_y'       :   'mouth_right_corner_y',
            'right_eye_center_x'        :   'left_eye_center_x',
            'right_eye_center_y'        :   'left_eye_center_y',
            'right_eye_inner_corner_x'  :   'left_eye_inner_corner_x',
            'right_eye_inner_corner_y'  :   'left_eye_inner_corner_y',
            'right_eye_outer_corner_x'  :   'left_eye_outer_corner_x',
            'right_eye_outer_corner_y'  :   'left_eye_outer_corner_y',
            'right_eyebrow_inner_end_x' :   'left_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y' :   'left_eyebrow_inner_end_y',
            'right_eyebrow_outer_end_x' :   'left_eyebrow_outer_end_x',
            'right_eyebrow_outer_end_y' :   'left_eyebrow_outer_end_y',
            'mouth_right_corner_x'      :   'mouth_left_corner_x',
            'mouth_right_corner_y'      :   'mouth_left_corner_y'
        }

        return Y.apply(lambda keypoint : Y[columns_mapping[keypoint.name]].values if columns_mapping.has_key(keypoint.name) else keypoint.values) \
                .apply(lambda keypoint : 96 - keypoint if keypoint.name.endswith("_x") else keypoint)
    
    X_flipped = images2d_to_dataframe(flip_2d_images_horizontally(load2d()))
    Y_flipped = flip_labels_horizontally()

    return pd.concat([X, X_flipped]), pd.concat([Y, Y_flipped])


def df_predict(filename = TEST_FILE):
    """the test set provided by kaggle. """

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
