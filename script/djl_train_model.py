#!/usr/bin/python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect,codecs,sys
import time

import numpy as np
import tensorflow as tf

from djl_model_lstm import PTBInput
from djl_model_lstm import PTBModel
from config import get_train_config  as config
import djl_reader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("vocab_file", None,
                    "Where the vocab data is stored.")
flags.DEFINE_string("train_file", None,
                    "Where the training data is stored.")

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

def run_epoch(session, model,id_to_word = None, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)

	fetches = {
		"softmax_w": model.softmax_w,
		"output": model.output,
		"logits": model.logits,
		"targets": model.targets,
		"input": model.input.input_data,
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None: fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]
		input = [id_to_word[x] for x in vals["input"].reshape(-1)];
		targets = [id_to_word[x] for x in vals["targets"].reshape(-1)];
		print('loginfo:-----------------------------')
		print('input:' + ' '.join(input))
		print('target:' + ' '.join(targets))
#		print('output:',vals['output'])
#		print('prect:',np.fabs(vals['logits']))
		print('cost:',cost);
		costs += cost
		iters += model.input.num_steps
		if verbose == False and not id_to_word is None:
			wid = vals["input"][0][0];
			maxids = vals["logits"].argmax(axis = 1);
			print('input:',wid,id_to_word[wid],' prect:',maxids[0],id_to_word[maxids[0]])
	return np.exp(costs / iters)

def main(_):
	if not FLAGS.vocab_file:
		raise ValueError("Must set --vocab_file to PTB data directory");
	if not FLAGS.train_file:
		raise ValueError("Must set --train_file to PTB data directory");

	raw_data = reader.djl_raw_data(FLAGS.vocab_file,FLAGS.train_file);
	train_data, train_tag, word_to_id, id_to_word = raw_data;

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

	with tf.name_scope("Train"):
		train_input = PTBInput(config=config, idata=train_data, tdata=train_tag, name="TrainInput")
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			m = PTBModel(is_training=True, config=config, input_=train_input)
		tf.summary.scalar("Training Loss", m.cost)
		tf.summary.scalar("Learning Rate", m.lr)

	sv = tf.train.Supervisor(logdir=FLAGS.save_path)
	with sv.managed_session() as session:
		for i in range(config.max_max_epoch):
			lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
			m.assign_lr(session, config.learning_rate * lr_decay)

			print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
			train_perplexity = run_epoch(session, m,id_to_word=id_to_word, eval_op=m.train_op,verbose=True)
			print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

		if FLAGS.save_path:
			print("Saving model to %s." % FLAGS.save_path)
			sv.saver.save(session, FLAGS.save_path + 'finalmodel', global_step=sv.global_step)

if __name__ == "__main__":
	tf.app.run()
