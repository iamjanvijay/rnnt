from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2
import tensorflow as tf

LOG_0 = -4000.0


def extract_diagonals(log_probs):
    time_steps = tf.shape(log_probs)[1]  # T
    output_steps = tf.shape(log_probs)[2]  # U + 1
    reverse_log_probs = tf.reverse(log_probs, axis=[-1])
    paddings = [[0, 0], [0, 0], [time_steps - 1, 0]]
    padded_reverse_log_probs = tf.pad(reverse_log_probs, paddings,
                                      'CONSTANT', constant_values=LOG_0)
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

    alpha = tf.transpose(
        tf.concat([tf.expand_dims(initial_alpha, axis=0), fwd], axis=0), perm=[1, 2, 0])
    alpha = matrix_diag_part_v2(alpha, k=(0, target_max_len - 1), padding_value=LOG_0)
    alpha = tf.transpose(tf.reverse(alpha, axis=[1]), perm=[0, 2, 1])

    return alpha


def backward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len, label_length, logit_length, blank_sl):
    """
        :return: backward variable beta with shape batch_size x input_max_len x target_max_len
    """

    def next_state(x, mask_and_trans_probs):
        mask_s, blank_probs_s, truth_probs = mask_and_trans_probs

        beta_b = tf.concat([x[:, 1:] + blank_probs_s, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1)
        beta_t = tf.concat([x[:, :-1] + truth_probs, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1)

        beta_next = tf.reduce_logsumexp(tf.stack([beta_b, beta_t], axis=0), axis=0)
        masked_beta_next = beta_next * tf.expand_dims(mask_s, axis=1) + x * tf.expand_dims((1.0 - mask_s), axis=1)
        return masked_beta_next

    # Initial beta for batches.
    initial_beta_mask = tf.one_hot(logit_length-1, depth=input_max_len+1)
    initial_beta = tf.expand_dims(blank_sl, axis=1) * initial_beta_mask + LOG_0 * (1.0 - initial_beta_mask)

    # Mask for scan iterations.
    mask = tf.sequence_mask(logit_length+label_length-1, input_max_len+target_max_len-2, dtype=tf.dtypes.float32)
    mask = tf.transpose(mask, perm=[1, 0])

    bwd = tf.scan(next_state, (mask, bp_diags[:-1, :, :], tp_diags), initializer=initial_beta, reverse=True)

    beta = tf.transpose(tf.concat([bwd, tf.expand_dims(initial_beta, axis=0)], axis=0), perm=[1, 2, 0])[:, :-1, :]
    beta = matrix_diag_part_v2(beta, k=(0, target_max_len - 1), padding_value=LOG_0)
    beta = tf.transpose(tf.reverse(beta, axis=[1]), perm=[0, 2, 1])

    return beta


def compute_rnnt_loss_and_grad_helper(logits, labels, label_length, logit_length):
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

    label_mask = tf.expand_dims(tf.sequence_mask(
        label_length + 1, maxlen=target_max_len, dtype=tf.float32), axis=1)
    input_mask = tf.expand_dims(tf.sequence_mask(
        logit_length, maxlen=input_max_len, dtype=tf.float32), axis=2)

    mask = label_mask * input_mask
    mask_log_values = (1-mask) * LOG_0

    alpha = forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len) * mask + mask_log_values

    indices = tf.stack([logit_length - 1, label_length], axis=1)
    blank_sl = tf.gather_nd(blank_probs, indices, batch_dims=1)
    beta = backward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len, label_length, logit_length, blank_sl) * mask + mask_log_values
    final_state_probs = beta[:, 0, 0]

    tiled_fsp = tf.tile(
        tf.reshape(final_state_probs, shape=[batch_size, 1, 1]), multiples=[1, input_max_len, target_max_len]) * mask + mask_log_values

    sl_indices = tf.stack([tf.range(0, batch_size, dtype=tf.int64), logit_length-1, label_length], axis=1)
    sl_one_hot = tf.scatter_nd(sl_indices, tf.ones(shape=batch_size), shape=alpha.shape)

    beta_shifted_u = tf.concat([beta[:, :, 1:], tf.ones(shape=[batch_size, input_max_len, 1])*LOG_0], axis=2)
    beta_shifted_t = tf.concat([beta[:, 1:, :], tf.ones(shape=[batch_size, 1, target_max_len])*LOG_0], axis=1)
    beta_shifted_t -= sl_one_hot * LOG_0

    grads_tp = (alpha - tiled_fsp + beta_shifted_u) * mask + mask_log_values
    grads_bp = alpha - tiled_fsp + beta_shifted_t

    truth_mask = tf.concat([one_hot_labels, tf.zeros(shape=[batch_size, input_max_len, 1, vocab_size])], axis=2)

    grads_p = tf.ones(shape=log_probs.shape) * LOG_0 * (1-truth_mask) + tf.expand_dims(grads_tp, axis=-1) * truth_mask
    grads_p = tf.concat([tf.expand_dims(grads_bp, axis=-1), grads_p[:, :, :, 1:]], axis=-1)

    dpp_dlp = blank_probs + tf.math.log(1 - tf.exp(blank_probs))
    ndpp_dli = tf.expand_dims(blank_probs, axis=-1) + log_probs[:, :, :, 1:]

    expanded_truth_probs = tf.expand_dims(tf.concat([truth_probs, tf.ones(shape=[batch_size, input_max_len, 1])*LOG_0], axis=2), axis=-1)

    dpy_dly = (expanded_truth_probs + tf.math.log(1 - tf.exp(expanded_truth_probs))) * truth_mask
    ndpy_dli = expanded_truth_probs + log_probs

    dl_dp = -tf.exp(grads_bp + dpp_dlp) + tf.exp(grads_tp + ndpy_dli[:, :, :, 0])
    dl_dy = (tf.exp(tf.expand_dims(grads_bp, axis=-1) + ndpp_dli) - tf.exp(grads_p[:, :, :, 1:] + dpy_dly[:, :, :, 1:])) * truth_mask[:, :, :, 1:]
    dl_di = (tf.exp(tf.expand_dims(grads_bp, axis=-1) + ndpp_dli) + tf.exp(tf.expand_dims(grads_tp, axis=-1) + ndpy_dli[:, :, :, 1:])) * (1 - truth_mask[:, :, :, 1:])

    grads = tf.concat([tf.expand_dims(dl_dp, -1), dl_dy + dl_di], axis=-1)

    loss = -beta[:, 0, 0]
    return loss, grads


def rnnt_loss(logits, labels, label_length, logit_length, name=None):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        args = [logits, labels, label_length, logit_length]

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t, labels_t, label_length_t, logit_length_t):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(logits=logits_t, labels=labels_t, label_length=label_length_t, logit_length=logit_length_t)
            result = compute_rnnt_loss_and_grad_helper(**kwargs)

            def grad(grad_loss):
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], grad

        return compute_rnnt_loss_and_grad(*args)
