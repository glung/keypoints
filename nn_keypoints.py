# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the facial keypoints detection network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy

# The facial keypoints detection dataset has 30 classes,
# representing 30 coordinates for 15 keypoints.
NUM_OUTPUT = 30

# The facial keypoints detection images are always 96x96 pixels.
IMAGE_SIZE = 96
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images):
    """Build the facial keypoints detection model up to where it may be used for inference.
    """

    with tf.name_scope('perceptron'):
        # NOTE : maybe it does not converge because classes cannot share
        # weights
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, NUM_OUTPUT],
                stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS))), 
            name = 'weights'
        )
        biases = tf.Variable(tf.zeros([NUM_OUTPUT]), name = 'biases')
        return tf.matmul(images, weights) + biases


def loss(y_, y):
    """Calculates the loss from the logits and the labels.

    """

    tfLoss = tf.sqrt(tf.reduce_sum(tf.square(y - y_)))
    #tfLoss = tf.reduce_mean(tf.square(y - y_)) ->  was going to INFINITY
    # tfLoss = tf.Print(tfLoss, [tfLoss], "LOSS:")
    return tfLoss


def training(loss, learning_rate):
    """Sets up the training Ops.

    """
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    return optimizer.minimize(loss, global_step = global_step)


def evaluation(y_, y):
    """Evaluate the quality of the regression

    """

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    return cross_entropy
