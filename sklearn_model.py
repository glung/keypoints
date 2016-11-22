from sklearn import linear_model
from sklearn import pipeline as skp
from sklearn import preprocessing


def linearRegression():
    """construct the model. """

    print ("sklearn linearRegression")

    return linear_model.LinearRegression(fit_intercept=False)


def linearRige():
    """construct the model. """

    print ("sklearn Ridge ")

    return skp.Pipeline([
        # ('preprocess', self.preprocessing()),
        ('regressor', linear_model.Ridge(alpha = 10000))
    ])

