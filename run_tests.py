import os
import time
import numpy as np
import tensorflow as tf

# import warprnnt_tensorflow
from rnnt_loss import compute_rnnt_loss_and_grad_helper, rnnt_loss

EPS = 0.001


def loss_grad_gradtape(logits, labels, label_lengths, logit_lengths):
    with tf.GradientTape() as g:
        g.watch(logits)
        loss = rnnt_loss(logits, labels, label_lengths, logit_lengths)
    grad = g.gradient(loss, logits)
    return loss, grad


@tf.function
def tf_loss(logits, labels, label_lengths, logit_lengths):
    loss = rnnt_loss(logits, labels, label_lengths, logit_lengths)
    return loss


@tf.function
def warp_loss(logits, labels, label_lengths, logit_lengths):
   log_probs = tf.nn.log_softmax(logits, axis=3)
   loss = warprnnt_tensorflow.rnnt_loss(
       log_probs, labels, logit_lengths, label_lengths)
   return loss


if __name__ == '__main__':
    test_cases_dir = 'test_scripts/testcases'
    test_case_results = dict()
    for subdir, direc, files in os.walk(test_cases_dir):
        for filename in files:
            if not filename.endswith('.npy'):
                continue
            base_folder = subdir.strip().split('/')[-1]
            file_path = os.path.join(subdir, filename)
            data = np.load(file_path, allow_pickle=True).item()
            logits = tf.convert_to_tensor(data['acts'])
            labels = tf.convert_to_tensor(data['labels'])
            label_lengths = tf.convert_to_tensor(data['label_lengths'])
            logit_lengths = tf.convert_to_tensor(data['input_lengths'])
            true_loss = np.array(data['final_loss'])
            true_grads = np.array(data['grads'])

            # Testing the helper function directly.
            pred_loss, pred_grads = compute_rnnt_loss_and_grad_helper(
                logits, labels, label_lengths, logit_lengths)

            error_loss = np.abs(true_loss - pred_loss)
            error_grads = np.abs(true_grads - pred_grads)

            smape_loss = np.mean(np.nan_to_num(error_loss/(0.5*(np.abs(true_loss) + np.abs(pred_loss)))))
            smape_grads = np.mean(np.nan_to_num(error_grads/(0.5*(np.abs(true_grads) + np.abs(pred_grads)))))

            hfunc_passed_loss = smape_loss < EPS
            hfunc_passed_grads = smape_grads < EPS

            print('hfunc', file_path, np.max(error_loss), np.max(error_grads), hfunc_passed_loss, hfunc_passed_grads)

            # Testing the final function under TF-2.0 Gradient-Tape.
            pred_loss, pred_grads = loss_grad_gradtape(
                logits, labels, label_lengths, logit_lengths)

            error_loss = np.abs(true_loss - pred_loss)
            error_grads = np.abs(true_grads - pred_grads)

            smape_loss = np.mean(np.nan_to_num(error_loss/(0.5*(np.abs(true_loss) + np.abs(pred_loss)))))
            smape_grads = np.mean(np.nan_to_num(error_grads/(0.5*(np.abs(true_grads) + np.abs(pred_grads)))))

            gtape_passed_loss = smape_loss < EPS
            gtape_passed_grads = smape_grads < EPS

            print('gtape', file_path, np.max(error_loss), np.max(error_grads), gtape_passed_loss, gtape_passed_grads)

            # Testing for performance speed.
            labels_32, label_lengths_32, logit_lengths_32 = tf.dtypes.cast(labels, tf.int32), tf.dtypes.cast(label_lengths, tf.int32), tf.dtypes.cast(logit_lengths, tf.int32)
            st = time.time()
            # warp_loss_val = warp_loss(logits, labels_32, label_lengths_32, logit_lengths_32)
            warp_time = time.time()-st
            st = time.time()
            # tf_loss_val = tf_loss(logits, labels, label_lengths, logit_lengths)
            tf_time = time.time()-st

            # print('warp', file_path, warp_time)
            # print('tf', file_path, tf_time)

            # Each base folder will have list of tuples for the testcases - [(test_case_name, hfunc_passed_loss, hfunc_passed_grads, gtape_passed_loss, gtape_passed_grads, warp_time, tf_time) ...]
            if base_folder not in test_case_results:
                test_case_results[base_folder] = [] 
            test_case_results[base_folder].append((filename, hfunc_passed_loss, hfunc_passed_grads, gtape_passed_loss, gtape_passed_grads, warp_time, tf_time))

    for base_folder in test_case_results:
        hfunc_passed = 0
        gtape_passed = 0
        avg_warp_time = 0
        avg_tf_time = 0
        num_testcases = 0
        for test_case_tuple in test_case_results[base_folder]:
            filename, hfunc_passed_loss, hfunc_passed_grads, gtape_passed_loss, gtape_passed_grads, warp_time, tf_time = test_case_tuple
            num_testcases += 1
            hfunc_passed += (1 if (hfunc_passed_loss and hfunc_passed_grads) else 0)
            gtape_passed += (1 if (gtape_passed_loss and gtape_passed_grads) else 0)
            avg_warp_time += warp_time
            avg_tf_time += tf_time
        print("Type - {} | H-Func Passed - {}/{} | GTape Passed - {}/{} | Avg Warp Time - {} ms | Avg TF Time - {} ms".format(base_folder, hfunc_passed, num_testcases, gtape_passed, num_testcases, 1000.0*avg_warp_time/num_testcases, 1000.0*avg_tf_time/num_testcases))
