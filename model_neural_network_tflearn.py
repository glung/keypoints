import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing


def model_one_hidden_layer():
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=121.105057897)
    img_prep.add_featurewise_stdnorm(std=59.2330252646)

    # TODO : shape may not be working properly with the notebook setup
    # Update the tflearn version and try to use the placeholder above
    # https://github.com/tflearn/tflearn/issues/360
    # http://stackoverflow.com/questions/40917017/typeerror-using-a-tf-tensor-as-a-python-bool-is-not-allowed

    # TODO: to save time, dump the input data in a binary file?

    network = input_data(shape=[None, (96 * 96)], data_preprocessing=img_prep)
    network = fully_connected(network, 100)
    network = fully_connected(network, 30)
    # sgd = tflearn.SGD(learning_rate=learning_rate)
    momentum = tflearn.Momentum(learning_rate=0.01, decay_step=100)
    network = regression(network, optimizer=momentum, loss='mean_square')

    return tflearn.DNN(network)


def model_conv_net():
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    network = input_data(shape=[96, 96, 1], data_preprocessing=img_prep, name='input')
    # Do we need a L2 regularizer ?
    # How to initialize variables ?
    # Do we need a drop out ?
    # Do we need strides?
    # Is AdamOptimizer better ?
    network = conv_2d(network, 32, 3)  # -> 96 * 96 * 32
    network = max_pool_2d(network, 2)  # -> 48 * 48 * 32
    network = conv_2d(network, 64, 2)  # -> 48 * 48 * 64
    network = max_pool_2d(network, 2)  # -> 24* 24 * 64
    network = conv_2d(network, 128, 2)  # -> 24* 24 * 128
    network = max_pool_2d(network, 2)  # -> 12* 12 * 128
    network = fully_connected(network, 500)
    network = fully_connected(network, 500)
    network = fully_connected(network, 30)

    # momentum = tflearn.Momentum(learning_rate=0.01, decay_step=100)
    adam = tflearn.Adam(learning_rate=0.001)
    network = regression(network, optimizer=adam, loss='mean_square')

    return tflearn.DNN(network,
                       tensorboard_dir="target/tflearn_logs/", tensorboard_verbose=0,
                       # checkpoint_path="target/model/checkpoints/"
                       )
