# RNN-Transducer Loss
This package provides a implementation of Transducer Loss in TensorFlow==2.0.

## Using the pakage
First install the module using pip command.
```bash
pip install rnnt
```
Then use the "rnnt" loss funtion from "rnnt" module, as described in the sample script: [Sample Train Script](https://github.com/iamjanvijay/rnnt/blob/master/source/sample_train.py)
```python
from rnnt import rnnt_loss

def loss_grad_gradtape(logits, labels, label_lengths, logit_lengths):
    with tf.GradientTape() as g:
        g.watch(logits)
        loss = rnnt_loss(logits, labels, label_lengths, logit_lengths)
    grad = g.gradient(loss, logits)
    return loss, grad
    
pred_loss, pred_grads = loss_grad_gradtape(logits, labels, label_lengths, logit_lengths)
```
Follwing are the shapes of input parameters for rnnt_loss method - <br>
logits - (batch_size, input_time_steps, output_time_steps+1, vocab_size+1) <br>
labels - (batch_size, output_time_steps) <br>
label_length - (batch_size) - number of time steps for each output sequence in the minibatch. <br>
logit_length - (batch_size) - number of time steps for each input sequence in the minibatch.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{singh2020rnnt-loss,
  title        = {RNNT-Loss},
  author       = {Singh, Janvijay},
  year         = {2020},
  archivePrefix = {arXiv}
}
```
