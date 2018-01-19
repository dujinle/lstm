#!/usr/bin/python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class get_train_config():
	"""Small config."""
	init_scale = 0.1
	learning_rate = 0.95
	max_grad_norm = 5
	num_layers = 2
	num_steps = 10
	hidden_size = 1000
	max_epoch = 5
	max_max_epoch = 10
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 10
	vocab_size = 6000

class get_prect_config():
	"""Small config."""
	init_scale = 0.1
	learning_rate = 0.95
	max_grad_norm = 5
	num_layers = 2
	num_steps = 1
	hidden_size = 1000
	max_epoch = 1
	max_max_epoch = 2
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 1
	vocab_size = 6000

