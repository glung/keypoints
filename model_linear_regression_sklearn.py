from sklearn import linear_model
from sklearn import pipeline as skp


def model_linear_regression():
    """construct the model. """

    return linear_model.LinearRegression()


def model_linear_rige():
    """construct the model. """

    return skp.Pipeline([
        # ('preprocess', self.preprocessing()),
        ('regressor', linear_model.Ridge(alpha = 10000))
    ])

