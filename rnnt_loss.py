from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def transition_probs(labels, log_probs):
    blank_probs = log_probs[:, :, :, 0]
    labels = tf.one_hot(tf.tile(tf.expand_dims(labels, axis=1),
                                multiples=[1, log_probs.shape[1], 1]), depth=log_probs.shape[-1])
    truth_probs = tf.reduce_sum(tf.multiply(log_probs, labels), axis=-1)

    return blank_probs, truth_probs
