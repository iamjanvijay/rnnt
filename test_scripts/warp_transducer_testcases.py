import warprnnt_tensorflow
import numpy as np
import tensorflow as tf

def gen_test_case(batch_num, max_label_length, max_input_length, output_vocab_size):
	# Inputs
	label_lengths = np.random.randint(low=1, high=max_label_length+1, size=(batch_num))
	blank_label = 0 # assuming zero for now; np.random.randint(low=0, high=output_vocab_size+1, size=())
	input_lengths = np.random.randint(low=1, high=max_input_length+1, size=(batch_num))
	labels = np.random.randint(low=1, high=output_vocab_size+1, size=(batch_num, max_label_length))
	acts = np.random.rand(batch_num, max_input_length, max_label_length, output_vocab_size+1)
	log_probs = tf.nn.log_softmax(acts, axis=3)
	
	# Outputs
	final_loss = warprnnt_tensorflow.rnnt_loss(acts, labels, input_lengths, label_lengths, blank_label)

	return {'acts': acts, 'log_probs': log_probs, 'labels': labels, 'input_lengths': input_lengths, 'label_lengths': label_lengths, 'blank_label': blank_label, 'final_loss': final_loss}
	
if __name__ == '__main__':
	batch_num = 1
	max_label_length = 5
	max_input_length = 5
	output_vocab_size = 3 # Without blank symbol
	num_test_cases = 5
	
	for i in range(num_test_cases):
		test_case = gen_test_case(batch_num, max_label_length, max_input_length, output_vocab_size)
		np.save('test_case_{}.npy'.format(i), test_case)


	
