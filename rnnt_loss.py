from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2
import tensorflow as tf

LOG_0 = -100.0


def extract_diagonals(log_probs):
    time_steps = tf.shape(log_probs)[1]  # T
    output_steps = tf.shape(log_probs)[2]  # U + 1
    reverse_log_probs = tf.reverse(log_probs, axis=[-1])
    paddings = [[0, 0], [0, 0], [time_steps - 1, 0]]
    padded_reverse_log_probs = tf.pad(reverse_log_probs, paddings, 'CONSTANT', constant_values=LOG_0)
    diagonals = matrix_diag_part_v2(padded_reverse_log_probs, k=(0, time_steps + output_steps - 2),
                                    padding_value=LOG_0)

    return tf.transpose(diagonals, perm=[1, 0, 2])


def transition_probs(one_hot_labels, log_probs):
    """
    :return: blank_probs with shape batch_size x input_max_len x target_max_len
             truth_probs with shape batch_size x input_max_len x (target_max_len-1)
    """
    blank_probs = log_probs[:, :, :, 0]
    truth_probs = tf.reduce_sum(tf.multiply(log_probs[:, :, :-1, :], one_hot_labels), axis=-1)

    return blank_probs, truth_probs


def forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len):
    """
    :return: forward variable alpha with shape batch_size x input_max_len x target_max_len
    """

    def next_state(x, trans_probs):
        blank_probs = trans_probs[0]
        truth_probs = trans_probs[1]

        x_b = tf.concat([LOG_0 * tf.ones(shape=[batch_size, 1]), x[:, :-1] + blank_probs], axis=1)
        x_t = x + truth_probs

        x = tf.reduce_logsumexp(tf.stack([x_b, x_t], axis=0), axis=0)
        return x

    initial_alpha = tf.concat(
        [tf.zeros(shape=[batch_size, 1]), tf.ones(shape=[batch_size, input_max_len - 1]) * LOG_0], axis=1)

    fwd = tf.scan(next_state, (bp_diags[:-1, :, :-1], tp_diags), initializer=initial_alpha)

    alpha = tf.transpose(tf.concat([tf.expand_dims(initial_alpha, axis=0), fwd], axis=0), perm=[1, 2, 0])
    alpha = matrix_diag_part_v2(alpha, k=(0, target_max_len - 1), padding_value=LOG_0)
    alpha = tf.transpose(tf.reverse(alpha, axis=[1]), perm=[0, 2, 1])

    return alpha


def backward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len):
    """
        :return: backward variable beta with shape batch_size x input_max_len x target_max_len
    """
    # TODO: implement this to return beta with shape batch_size x input_max_len x target_max_len
    return tf.zeros(shape=[batch_size, input_max_len, target_max_len])


def rnnt_loss_and_grad(logits, labels, label_length, logit_length):
    batch_size = logits.shape[0]
    input_max_len = logits.shape[1]
    target_max_len = logits.shape[2]
    vocab_size = logits.shape[3]

    one_hot_labels = tf.one_hot(tf.tile(tf.expand_dims(labels, axis=1),
                                multiples=[1, input_max_len, 1]), depth=vocab_size)

    log_probs = tf.nn.log_softmax(logits)
    blank_probs, truth_probs = transition_probs(one_hot_labels, log_probs)
    bp_diags = extract_diagonals(blank_probs)
    tp_diags = extract_diagonals(truth_probs)

    label_mask = tf.expand_dims(tf.sequence_mask(label_length+1, maxlen=target_max_len, dtype=tf.float32), axis=1)
    input_mask = tf.expand_dims(tf.sequence_mask(logit_length, maxlen=input_max_len, dtype=tf.float32), axis=2)
    mask = label_mask * input_mask

    alpha = forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len) * mask

    indices = tf.stack([logit_length - 1, label_length], axis=1)
    alpha_sl = tf.gather_nd(alpha, indices, batch_dims=1)
    final_state_probs = alpha_sl + tf.gather_nd(blank_probs, indices, batch_dims=1)

    beta = backward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len) * mask

    tiled_fsp = tf.tile(
        tf.reshape(final_state_probs, shape=[batch_size, 1, 1]), multiples=[1, input_max_len, target_max_len])

    grads_blank = tiled_fsp - alpha - beta
    grads_blank_logits = tf.exp(grads_blank + blank_probs + tf.math.log(1 - tf.exp(blank_probs))) \
        * tf.concat([mask[:, 1:, :], tf.zeros(shape=[batch_size, 1, target_max_len])], axis=1)
    grads_blank_logits += tf.scatter_nd(tf.stack([tf.range(0, batch_size, dtype=tf.int64), logit_length-1, label_length], axis=1), alpha_sl, shape=grads_blank_logits.shape)

    grads_truth = (tiled_fsp[:, :, :-1] - alpha[:, :, :-1] - beta[:, :, 1:])
    grads_truth_logits = tf.exp(grads_truth + truth_probs + tf.math.log(1 - tf.exp(truth_probs))) * mask[:, :, 1:]

    grads_logits = (tf.expand_dims(grads_truth_logits, axis=-1) * one_hot_labels)[:, :, :, 1:]
    grads_logits = tf.concat([tf.expand_dims(grads_blank_logits, axis=-1),
                              tf.concat([grads_logits,
                                         tf.zeros(shape=[batch_size, input_max_len, 1, vocab_size-1])], axis=2)], axis=-1)

    loss = -final_state_probs
    return loss, grads_logits
