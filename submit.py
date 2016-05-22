# -*- coding: utf-8 -*-

"""
    submit
    ~~~~~~

    submit challenge

    :license: BSD
"""

from collections import OrderedDict

import pandas as pd

def submission(Y, lookup):
    """
    convert to the right format for submission. joins id lookup table to
    the predictions like this:

    prediction: (imageid, featureid, location) ­ unstacks first.
    lookup: (rowid, imageid, featureid)
    joined: (rowid, location)
    """

    predictions = unstacked_predictions(Y)
    joined = lookup.join(predictions).reset_index()

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
