#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
Prect the model described in:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect,codecs,sys,os
import time

import numpy as np
import tensorflow as tf

base_path = os.path.dirname(__file__);
sys.path.append(os.path.join(base_path,'.'));

from djl_model_lstm import PTBInput
from djl_model_lstm import PTBModel
from config import get_prect_config as config
import djl_reader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string("vocab_path", None,
                    "Where the vocab data is stored.")

flags.DEFINE_string("test_path", None,
                    "Where the test data is stored.")
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

	fp = codecs.open(FLAGS.test_path + '.pret','w','utf8');
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
			fp.write(id_to_word[wid] + id_to_word[maxids[0]]);
			if iters % 20 == 0:
				fp.write('\n');
	fp.close();
	return np.exp(costs / iters)

def main(_):
	if not FLAGS.test_path:
		raise ValueError("Must set --test_path to PTB data directory")

	if not FLAGS.vocab_path:
		raise ValueError("Must set --vocab_path to PTB data directory")

	raw_data = reader.djl_raw_data(FLAGS.vocab_path,FLAGS.test_path)
	test_data, test_tag,word_to_id,id_to_word = raw_data

	with tf.name_scope("Test"):
		test_input = PTBInput(config=config, idata=test_data, tdata=test_tag, name="TestInput")
		with tf.variable_scope("Model"):
			mtest = PTBModel(is_training=False, config=config,input_=test_input)

	sv = tf.train.Supervisor(logdir=FLAGS.save_path)
	with sv.managed_session() as session:
		if FLAGS.save_path:
			model_path = tf.train.latest_checkpoint(FLAGS.save_path);
			print(model_path);
			load_path = sv.saver.restore(session,model_path);

		test_perplexity = run_epoch(session, mtest,id_to_word= id_to_word)
		print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
	tf.app.run()
