#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect,codecs,sys
import time

import numpy as np
import tensorflow as tf

import djl_reader as reader

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS

sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
	"""The input data."""
	def __init__(self, config, idata,tdata,name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(idata) // batch_size) - 1) // num_steps
		self.input_data,self.targets = reader.djl_producer(idata,tdata, batch_size, num_steps, name=name)

class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._input = input_

		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		# Slightly better results can be obtained with forget gate biases
		# initialized to 1 but the hyperparameters of the model would need to be
		# different than reported in the paper.
		def lstm_cell():
		# With the latest TensorFlow source code (as of Mar 27, 2017),
		# the BasicLSTMCell will need a reuse parameter which is unfortunately not
		# defined in TensorFlow 1.0. To maintain backwards compatibility, we add
		# an argument check here:
			if 'reuse' in inspect.getargspec(
				tf.contrib.rnn.BasicLSTMCell.__init__).args:
				return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
			else:
				return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
		attn_cell = lstm_cell
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

		self._initial_state = cell.zero_state(batch_size, data_type())

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
			if is_training and config.keep_prob < 1:
				inputs = tf.nn.dropout(inputs, config.keep_prob)

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

		logits = tf.matmul(output, softmax_w) + softmax_b

		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[tf.reshape(input_.targets, [-1])],
			[tf.ones([batch_size * num_steps], dtype=data_type())])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state
		self._logits = logits
		self._softmax_w = softmax_w
		self._softmax_b = softmax_b
		self._output = output

		if not is_training: return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)


	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def logits(self): return self._logits

	@property
	def output(self): return self._output

	@property
	def softmax_w(self): return self._softmax_w

	@property
	def targets(self): return self._input.targets

	@property
	def logits(self): return self._logits

	@property
	def input(self): return self._input

	@property
	def initial_state(self): return self._initial_state

	@property
	def cost(self): return self._cost

	@property
	def final_state(self): return self._final_state

	@property
	def lr(self): return self._lr

	@property
	def train_op(self): return self._train_op

