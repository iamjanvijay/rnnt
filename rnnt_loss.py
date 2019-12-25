from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def extract_diagonals(log_probs):
	time_steps = tf.shape(log_probs)[1] # T
	output_steps = tf.shape(log_probs)[2] # U + 1
	reverse_log_probs = tf.reverse(log_probs, axis=[-1])
	paddings = [[0, 0], [0, 0], [time_steps-1, 0]]
	padded_reverse_log_probs = tf.pad(reverse_log_probs, paddings, 'CONSTANT', constant_values=tf.math.log(0.0))
	diagonals = tf.linalg.diag_part(padded_reverse_log_probs, k=(0, time_steps+output_steps-2), padding_value=tf.math.log(0.0))

	return diagonals


def transition_probs(labels, log_probs):
    blank_probs = log_probs[:, :, :, 0]
    labels = tf.one_hot(tf.tile(tf.expand_dims(labels, axis=1),
                                multiples=[1, log_probs.shape[1], 1]), depth=log_probs.shape[-1])
    truth_probs = tf.reduce_sum(tf.multiply(log_probs, labels), axis=-1)

    return blank_probs, truth_probs


def next_state(alpha, trans_probs):
    blank_probs = trans_probs[0]
    truth_probs = trans_probs[1]

    alpha_b = tf.concat([tf.math.log(tf.zeros(shape=[alpha.shape[0], 1])), alpha[:, :-1] + blank_probs], axis=1)
    alpha_t = tf.concat([alpha[:, :-1] + truth_probs, tf.math.log(tf.zeros(shape=[alpha.shape[0], 1]))], axis=1)

    alpha = tf.reduce_logsumexp(tf.stack([alpha_b, alpha_t], axis=0), axis=0)
    return alpha


def rnnt_loss_and_grad(logits, labels, label_length, logit_length):
    log_probs = tf.nn.log_softmax(logits)

    initial_alpha = tf.zeros(shape=[labels.shape[0], labels.shape[1]+1])
    initial_alpha[:, 0] = 1.0
    initial_alpha = tf.math.log(initial_alpha)

    trans_probs = transition_probs(labels, log_probs)

    fwd = tf.scan(next_state, trans_probs, initializer=initial_alpha)
    alpha = tf.transpose(tf.concat([initial_alpha, fwd], axis=0), perm=[1, 2, 0])

