import glob

import numpy as np
import tensorflow as tf

from rnnt_loss import compute_rnnt_loss_and_grad_helper, rnnt_loss

EPS = 0.0001


def loss_grad_gradtape(logits, labels, label_lengths, logit_lengths):
    with tf.GradientTape() as g:
        g.watch(logits)
        loss = rnnt_loss(logits, labels, label_lengths, logit_lengths)
    grad = g.gradient(loss, logits)
    return loss, grad


if __name__ == '__main__':
    hfunc_passed_loss = {}
    hfunc_passed_grads = {}
    gtape_passed_loss = {}
    gtape_passed_grads = {}
    test_files = glob.glob('test_scripts/test_case*.npy')
    for i in test_files:
        data = np.load(i, allow_pickle=True).item()
        logits = tf.convert_to_tensor(data['acts'])
        labels = tf.convert_to_tensor(data['labels'])
        label_lengths = tf.convert_to_tensor(data['label_lengths'])
        logit_lengths = tf.convert_to_tensor(data['input_lengths'])
        true_loss = np.array(data['final_loss'])
        true_grads = np.array(data['grads'])

        # Testing the helper function directly.
        pred_loss, pred_grads = compute_rnnt_loss_and_grad_helper(logits, labels, label_lengths, logit_lengths)
        hfunc_passed_loss[i] = (np.sum(np.abs(true_loss - pred_loss.numpy())) < EPS)
        hfunc_passed_grads[i] = (np.sum(np.abs(true_grads - pred_grads.numpy())) < EPS)

        # Testing the final function under TF-2.0 Gradient-Tape.
        pred_loss, pred_grads = loss_grad_gradtape(logits, labels, label_lengths, logit_lengths)
        gtape_passed_loss[i] = (np.sum(np.abs(true_loss - pred_loss.numpy())) < EPS)
        gtape_passed_grads[i] = (np.sum(np.abs(true_grads - pred_grads.numpy())) < EPS)

    print("Test results for helper function -")
    print(hfunc_passed_loss)
    print(hfunc_passed_grads)

    print("Test results for TF-2.0 Gradient-Tape function -")
    print(gtape_passed_loss)
    print(gtape_passed_grads)
