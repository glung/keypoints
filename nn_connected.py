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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import nn_keypoints
import data

class FLAGS:
    learning_rate = 0.00001
    max_steps = 1000
    batch_size = 100
    train_dir = "tf_dir"


IMAGE_PIXELS = 96 * 96
NUM_OUTPUT = 1


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape = (batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.float32, shape = (batch_size, NUM_OUTPUT))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.

    # TODO
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: np.reshape(labels_feed[:,0], (FLAGS.batch_size, NUM_OUTPUT))
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    steps_per_epoch = data_set._num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    scores = 0
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        scores += sess.run(eval_correct, feed_dict = feed_dict)

    rmses = np.sqrt(np.abs(scores))
    std = np.std(rmses)
    mean = rmses.mean()

    print('  Num examples: %d scores: %f mean: %f  std: %f' % (num_examples, scores, mean, std))


def run_training():
    """Train KEYPOINTS for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on keypoints.
    data_sets = data.read_data_sets()
    print("[data_sets] train: %d, validation %d, test: %d" % (data_sets.train._num_examples, data_sets.validation._num_examples, data_sets.test._num_examples))

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        outputs = nn_keypoints.inference(images_placeholder)
        loss = nn_keypoints.loss(outputs, labels_placeholder)
        train_op = nn_keypoints.training(loss, FLAGS.learning_rate)
        eval_correct = nn_keypoints.evaluation(outputs, labels_placeholder)
        sess = tf.Session()

        sess.run(tf.initialize_all_variables())

        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                     images_placeholder,
                                     labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)

            duration = time.time() - start_time

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                print('[EVAL] Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)

run_training()