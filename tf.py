# -*- coding: utf-8 -*-

"""
    keypoints
    ~~~~~~~~~

    learn facial keypoints.

    :license: BSD
"""

__version__ = '0.0.1'

import data
import model_neural_network_tflearn
import pickle
import numpy as np
import datetime


def main():
    """run the training. """

    data_set = load_data()

    convNet(
        get_data(data_set.train),
        get_data(data_set.validation)
    )


def one_hidden_layer(X, Xtest, Y, Ytest):
    epochs = 400
    model = model_neural_network_tflearn.model_one_hidden_layer()
    model.fit(X, Y, n_epoch=epochs, validation_set=(Xtest, Ytest), show_metric=False)
    model.save(modelFilePath("model_neural_network_tflearn", epochs))


def convNet((x_train, y_train), (x_validation, y_validation)):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_epoch = 1000
    model = model_neural_network_tflearn.model_conv_net()
    x_train = x_train.reshape(-1, 96, 96, 1)
    x_validation = x_validation.reshape(-1, 96, 96, 1)
    model.fit(x_train, y_train,
              n_epoch=n_epoch,
              validation_set=(x_validation, y_validation),
              show_metric=False,
              run_id=run_id)
    model.save(modelFilePath("conv_lenet5_adam", n_epoch, run_id))


def modelFilePath(name, n_epoch, run_id):
    return "target/model/" + name + "_epoch_" + str(n_epoch) + "_id_" + str(run_id) + ".tflearn"


def get_data(data_set):
    X = np.array(data_set.X)
    Y = np.array(data_set.Y)
    return X, Y


def load_data():
    # data_set_ = data.read_data_sets()
    # pickle.dump(data_set_, open("data_set_full.pkl", "wb"))
    data_set = pickle.load(open(dataPath("data_set_full.pkl"), "rb"))
    return data_set


def dataPath(name):
    return "target/data/" + name


if __name__ == '__main__':
    main()
