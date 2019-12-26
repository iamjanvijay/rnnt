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
        true_loss = np.array(data['final_loss'])
        pred_loss = np.array(rnnt_loss_and_grad(logits, labels, None, None))
        passed[i] = (np.sum(np.abs(true_loss - pred_loss)) < EPS)

    print(passed)