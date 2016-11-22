import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
import time

def contrib_linearRegressor():
    """construct the model. """

    print ("TF::learn.LinearRegressor")

    tf.logging.set_verbosity(tf.logging.INFO)

    regressor = learn.LinearRegressor(
        target_dimension=30,
        feature_columns=[tf.contrib.layers.real_valued_column(column_name='',
                                                              dimension=96 * 96,
                                                              dtype=tf.python.framework.dtypes.float32)],
        model_dir="/tpm/tf/LinearRegressor" + str(time.time()),
        config=learn.RunConfig(
            save_summary_steps=100,
            save_checkpoints_secs=60
        )
    )

    class MyRegressor():
        def __init__(self):
            pass

        def fit(self, x, y):
            regressor.fit(np.float32(x), np.float32(y), max_steps=2000)

        def predict(self, x):
            return regressor.predict(np.float32(x))

    return MyRegressor()


def vanilla_linear_regression():
    """Vanilla linear regression with TF (by hand)"""

    batch_size=None
    x = tf.placeholder("float", [batch_size, 96*96])
    y_ = tf.placeholder("float", [batch_size, 30])

    W = tf.Variable(tf.zeros([96*96, 30]))
    b = tf.Variable(tf.zeros([30]))
    y = tf.matmul(x, W) + b

    loss = tf.reduce_mean(tf.square(y - y_))

    train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss)

    class MyRegressor():
        def __init__(self):
            self.sess = tf.Session()

        def fit(self, a_x, a_y):
            print (a_x.shape)
            self.sess.run(tf.initialize_all_variables())

            max_iterations = 1000
            for iteration in range(max_iterations):
                np_x = np.asarray(a_x)
                np_y = np.asarray(a_y)

                loss_value = None
                b_size = 10
                extra_items = len(a_x) % b_size
                nb_batches = (len(a_x) - extra_items) / b_size

                # print "#itens:", len(a_x), ", #batches", nb_batches, ", extra_items:", extra_items

                for i in range(nb_batches):
                    start_idx = i * b_size
                    end_idx = (i + 1) * b_size - 1
                    # print ("batch start:", start_idx, ", stop:", end_idx)

                    x_input = np_x[start_idx:end_idx]
                    y_input = np_y[start_idx:end_idx]
                    loss_value, _ = self.sess.run([loss, train_step], feed_dict={x: x_input, y_: y_input})

                # TODO cross validate
                if (iteration + 1) % 100 == 0: print "Iteration ", iteration, "LOSS:", loss_value


        def predict(self, a_x):
            return self.sess.run(y, feed_dict={x: np.asarray(a_x[:batch_size])})

    return MyRegressor()


def vanilla_linear_regression_fit_intercept():
    """Vanilla linear regression with TF (by hand) with 0 centring of data"""

    batch_size=None
    x = tf.placeholder("float", [batch_size, 96*96])
    y_ = tf.placeholder("float", [batch_size, 30])

    W = tf.Variable(tf.zeros([96*96, 30]))
    y = tf.matmul(x, W)
    loss = tf.reduce_mean(tf.square(y - y_))

    train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss)

    def preprocess_data(X, y):
        X_offset = np.average(X, axis=0)
        X_centered = X - X_offset
        y_offset = np.average(y, axis=0)

        y_centered = y - y_offset

        return X_centered, y_centered, X_offset, y_offset


    class MyRegressor():
        def __init__(self):
            self.sess = tf.Session()
            self.y_with_intercept = None

        def fit(self, a_x, a_y):
            print (a_x.shape)
            self.sess.run(tf.initialize_all_variables())

            centered_X, centered_y, X_offset, y_offset = preprocess_data(a_x, a_y)

            max_iterations = 1000
            for iteration in range(max_iterations):
                np_x = np.asarray(a_x)
                np_y = np.asarray(a_y)

                loss_value = None
                b_size = 10
                extra_items = len(a_x) % b_size
                nb_batches = (len(a_x) - extra_items) / b_size

                # print "#itens:", len(a_x), ", #batches", nb_batches, ", extra_items:", extra_items

                for i in range(nb_batches):
                    start_idx = i * b_size
                    end_idx = (i + 1) * b_size - 1
                    # print ("batch start:", start_idx, ", stop:", end_idx)

                    x_input = np_x[start_idx:end_idx]
                    y_input = np_y[start_idx:end_idx]
                    loss_value, _ = self.sess.run([loss, train_step], feed_dict={x: x_input, y_: y_input})

                # TODO cross validate
                if (iteration + 1) % 100 == 0: print ("Iteration ", iteration, "LOSS:", loss_value)

            self.y_with_intercept = y + self.get_intercept(X_offset, y_offset)


        def get_intercept(self, X_offset, y_offset):
            coeff = self.sess.run(W)
            value = y_offset - np.dot(X_offset, coeff)
            return np.float32(value)


        def predict(self, a_x):
            return self.sess.run(self.y_with_intercept, feed_dict={x: np.asarray(a_x[:batch_size])})


    return MyRegressor()
