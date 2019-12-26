import glob

import numpy as np
import tensorflow as tf

from rnnt_loss import rnnt_loss_and_grad

EPS = 0.0001


if __name__ == '__main__':
    passed = {}
    test_files = glob.glob('test_scripts/test_case*.npy')
    for i in test_files:
        data = np.load(i, allow_pickle=True).item()
        logits = tf.convert_to_tensor(data['acts'])
        labels = tf.convert_to_tensor(data['labels'])
        label_lengths = tf.convert_to_tensor(data['label_lengths'])
        logit_lengths = tf.convert_to_tensor(data['input_lengths'])
        true_loss = np.array(data['final_loss'])
        pred_loss = np.array(rnnt_loss_and_grad(logits, labels, label_lengths, logit_lengths))
        passed[i] = (np.sum(np.abs(true_loss - pred_loss)) < EPS)

    print(passed)
