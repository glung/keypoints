import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import preprocessing

import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

import time


def one_hidden_layer():
    """ Perceptron w/ one hidden layer """
    """ /!\ This implementation does not converge """


    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    max_iterations = 1000
    learning_rate = 0.01
    image_size = 96 * 96
    nb_keypoints = 30
    n_hidden_1 = 100

    print (
        "learning_rate:", learning_rate,
        ", image_size:", image_size,
        ", hidden_1:", n_hidden_1,
        ", keypoints:", nb_keypoints
    )

    weights = {
        'h1': tf.Variable(tf.random_normal([image_size, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, nb_keypoints]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([nb_keypoints]))
    }

    # INPUT
    x = tf.placeholder("float", [None, image_size])
    y_ = tf.placeholder("float", [None, nb_keypoints])
    # MODEL
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.square(pred - y_))
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return PreprocessorAndBatchedRegressor(x, y_, train_step, loss, pred, max_iterations=max_iterations)


class PreprocessorAndBatchedRegressor():
    def __init__(self, x, y_, op_train, op_loss, op_predict, max_iterations=100):
        self.placeholder_x = x
        self.placeholder_y_ = y_
        self.op_train = op_train
        self.op_loss = op_loss
        self.op_predict = op_predict
        self.batch_size = 1000
        self.max_iterations = max_iterations
        # todo : use column feature to get this right ?
        self.x_scaler = preprocessing.StandardScaler(with_std=False)
        self.y_scaler = preprocessing.StandardScaler(with_std=False)
        self.sess = tf.Session()

    def fit(self, a_x, a_y):
        print (a_x.shape)
        self.sess.run(tf.initialize_all_variables())

        np_x = np.asarray(self.x_scaler.fit_transform(a_x))
        np_y = np.asarray(self.y_scaler.fit_transform(a_y))

        for iteration in range(self.max_iterations):
            for i in range(self.divide_in_batches(a_x)):
                x_input, y_input = self.next_batch(i, np_x, np_y)
                self.train(x_input, y_input)

            # TODO cross validate
            self.eventually_log_loss(iteration, x_input, y_input)

    def divide_in_batches(self, a_x):
        return int(round(float(len(a_x)) / self.batch_size))

    def next_batch(self, i, np_x, np_y):
        start_idx = i * self.batch_size

        end_idx = (i + 1) * self.batch_size - 1

        x_input = np_x[start_idx:end_idx]
        y_input = np_y[start_idx:end_idx]
        return x_input, y_input

    def train(self, x_input, y_input):
        self.sess.run(self.op_train, feed_dict=self.feed_train(x_input, y_input))

    def eventually_log_loss(self, iteration, x_input, y_input):
        if iteration == 0 or (iteration + 1) % 100 == 0:
            loss_value = self.sess.run(self.op_loss, feed_dict=self.feed_train(x_input, y_input))
            print "Iteration ", iteration, "LOSS:", loss_value

    def predict(self, a_x):
        centered_x = self.x_scaler.transform(a_x)
        predict = self.sess.run(self.op_predict, feed_dict={self.placeholder_x: np.asarray(centered_x)})
        return self.y_scaler.inverse_transform(predict)

    def feed_train(self, x_input, y_input):
        return {self.placeholder_x: x_input, self.placeholder_y_: y_input}
