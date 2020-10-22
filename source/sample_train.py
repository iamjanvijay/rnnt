import numpy as np
from tensorflow import keras
import tensorflow as tf
from rnnt import rnnt_loss

np.random.seed(0)

# Dummy Model definition
class DummyModel(keras.Model):
    def __init__(self, encoder_lstm_size, decoder_lstm_size, decoder_vocab_size):
        super(DummyModel, self).__init__()

        self.encoder_lstm = keras.layers.LSTM(encoder_lstm_size, return_sequences=True)
        self.decoder_lstm = keras.layers.LSTM(decoder_lstm_size, return_sequences=True)

        self.joint_dense = keras.layers.Dense(units=decoder_vocab_size+1, use_bias=True, activation='relu')

        self.loss = rnnt_loss

    def create_mask(self, seq_lens, max_steps):
        return tf.sequence_mask(seq_lens, max_steps)

    def encoder_forward(self, encoder_seqs, encoder_seqs_one_hot, encoder_lens, max_encoder_steps, encoder_vocab_size):
        encoder_seqs_mask = self.create_mask(encoder_lens, max_encoder_steps)
        return self.encoder_lstm(encoder_seqs_one_hot, mask=encoder_seqs_mask)

    def decoder_forward(self, decoder_seqs, decoder_seqs_one_hot, decoder_lens, max_decoder_steps, decoder_vocab_size):
        decoder_seqs_mask = self.create_mask(decoder_lens+1, max_decoder_steps+1)
        return self.decoder_lstm(decoder_seqs_one_hot, mask=decoder_seqs_mask)

    def joint_forward(self, encoder_logits, decoder_logits, max_encoder_steps, max_decoder_steps):
        encoder_logits = tf.tile(tf.expand_dims(encoder_logits, axis=2), [1, 1, max_decoder_steps+1, 1])
        decoder_logits = tf.tile(tf.expand_dims(decoder_logits, axis=1), [1, max_encoder_steps, 1, 1])
        return self.joint_dense(tf.concat([encoder_logits, decoder_logits], axis=-1))

    def call(self, encoder_seqs, encoder_seqs_one_hot, encoder_lens, max_encoder_steps, encoder_vocab_size, decoder_seqs, decoder_seqs_one_hot, decoder_lens, max_decoder_steps, decoder_vocab_size):
        encoder_logits = self.encoder_forward(encoder_seqs, encoder_seqs_one_hot, encoder_lens, max_encoder_steps, encoder_vocab_size) # B * T * encoder_lstm_size
        decoder_logits = self.decoder_forward(decoder_seqs, decoder_seqs_one_hot, decoder_lens, max_decoder_steps, decoder_vocab_size) # B * (U+1) * decoder_lstm_size
        joint_logits = self.joint_forward(encoder_logits, decoder_logits, max_encoder_steps, max_decoder_steps) # B * T * (U+1) * decoder_vocab_size
        log_probs = tf.nn.log_softmax(joint_logits, axis=3)
        return tf.reduce_mean(self.loss(log_probs, decoder_seqs+1, decoder_lens, encoder_lens))

# Dummy Batch Generator
def batch_generator(batch_size, max_encoder_steps, encoder_vocab_size, max_decoder_steps, decoder_vocab_size):
    while True:
        encoder_seqs, decoder_seqs = [], []
        encoder_lens, decoder_lens = [], []
        for batch_seq in range(batch_size):
            # Create a dummy encoder sequence
            cur_encoder_steps = np.random.randint(low=1, high=max_encoder_steps+1, size=(1,))[0]
            cur_encoder_seq = np.random.randint(low=0, high=encoder_vocab_size, size=(cur_encoder_steps))
            encoder_seqs.append(np.pad(cur_encoder_seq, (0, max_encoder_steps-cur_encoder_steps)))
            encoder_lens.append(cur_encoder_steps)
            # create a dummy decoder sequence
            cur_decoder_steps = np.random.randint(low=1, high=max_decoder_steps+1, size=(1,))[0]
            cur_decoder_seq = np.random.randint(low=0, high=decoder_vocab_size, size=(cur_decoder_steps))
            decoder_seqs.append(np.pad(cur_decoder_seq, (0, max_decoder_steps-cur_decoder_steps)))
            decoder_lens.append(cur_decoder_steps)
        encoder_seqs, decoder_seqs = np.array(encoder_seqs).astype(np.int64), np.array(decoder_seqs).astype(np.int64)
        encoder_lens, decoder_lens = np.array(encoder_lens).astype(np.int64), np.array(decoder_lens).astype(np.int64)
        yield [encoder_seqs, encoder_lens, decoder_seqs, decoder_lens]

def create_one_hot(seq_ids, max_val): 
    ''' 
        Assumes an integer array of B * T with values in range [0, max_val-1]
        Returns the one-hot representation of size B * T * max_val
    '''
    B, T = seq_ids.shape
    one_hot = seq_ids.reshape(-1)
    one_hot = np.eye(max_val)[one_hot]
    one_hot = one_hot.reshape([B, T, max_val]).astype(np.float32)
    return one_hot

if __name__=='__main__':
    batch_size, max_encoder_steps, encoder_vocab_size, max_decoder_steps, decoder_vocab_size = 2, 6, 3, 5, 4
    model = DummyModel(5, 5, decoder_vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    for batch in batch_generator(batch_size, max_encoder_steps, encoder_vocab_size, max_decoder_steps, decoder_vocab_size):
        encoder_seqs, encoder_lens, decoder_seqs, decoder_lens = batch
        # encoder one hot representation
        encoder_seqs_one_hot = create_one_hot(encoder_seqs, encoder_vocab_size)
        # decoder one hot representations
        npad = ((0, 0), (1, 0)) # Pad the B * U to create B * (U+1)
        decoder_seqs_one_hot = create_one_hot(np.pad(decoder_seqs, pad_width=npad), decoder_vocab_size)
        decoder_seqs_one_hot[:, 0] = 0. # timestep 0 is reseved for blank symbol, all zeros input
        # model call
        with tf.GradientTape() as g:
            loss_val = model(encoder_seqs, encoder_seqs_one_hot, encoder_lens, max_encoder_steps, encoder_vocab_size, decoder_seqs, decoder_seqs_one_hot, decoder_lens, max_decoder_steps, decoder_vocab_size)
        print("Loss:", loss_val)
        grads = g.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
