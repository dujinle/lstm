from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections,os,codecs
import tensorflow as tf


def _read_vocab(filename):
	with codecs.open(filename,'r','utf8') as fp:
		data = fp.readlines();
		word_to_id = dict();
		id_to_word = dict();
		for i in data:
			iarray = i.strip('\n\r').split();
			word_to_id[iarray[0]] = iarray[1];
			id_to_word[int(iarray[1])] = iarray[0];
		return (word_to_id,id_to_word);

def _read_words(filename):
	with codecs.open(filename,'r','utf-8') as f:
		datas = f.readlines();
		idx = 0;
		input_data = list();
		target_data = list();
		for i in datas:
			iarray = i.strip('\n\r').split();
			for iw in iarray:
				if idx % 2 == 0:
					input_data.append(int(iw));
				else:
					target_data.append(int(iw));
			idx = idx + 1;
		return (input_data,target_data);

def _test_read_words(filename):
	with codecs.open(filename,'r','utf-8') as f:
		datas = f.readlines();
		idx = 0;
		input_data = list();
		for i in datas:
			iarray = i.strip('\n\r').split();
			for iw in iarray:
				input_data.append(int(iw));
		return input_data;

def djl_raw_data(vocabfile,trainfile):

	print('start read ',vocabfile,trainfile);
	word_to_id,id_to_word = _read_vocab(vocabfile);
	train_data,target_data = _read_words(trainfile);
	return train_data, target_data,word_to_id,id_to_word;

def djl_test_data(vocabfile,testfile):
	print('start read ',vocabfile,testfile);
	word_to_id,id_to_word = _read_vocab(vocabfile);
	test_data = _test_read_words(testfile);
	return test_data,word_to_id,id_to_word;

def djl_producer(raw_data,targets, batch_size, num_steps, name=None):
	with tf.name_scope(name, "DJLProducer", [raw_data, batch_size, num_steps]):
		print('djl producer------------------------------------')
		print('batch_size:',batch_size,'num_steps',num_steps);
		print(' '.join([str(wid) for wid in raw_data[:10]]))
		print(' '.join([str(wid) for wid in targets[:10]]))
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
		tar_data = tf.convert_to_tensor(targets, name="tar_data", dtype=tf.int32)
		data_len = tf.size(raw_data)
		batch_len = data_len // batch_size
		data = tf.reshape(raw_data[0 : batch_size * batch_len],[batch_size, batch_len])
		tars = tf.reshape(tar_data[0 : batch_size * batch_len],[batch_size, batch_len])

		epoch_size = (batch_len - 1) // num_steps
		assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = tf.strided_slice(data, [0, i * num_steps],[batch_size, (i + 1) * num_steps])
		x.set_shape([batch_size, num_steps])
		y = tf.strided_slice(tars, [0, i * num_steps],[batch_size, (i + 1) * num_steps])
		y.set_shape([batch_size, num_steps])
		return x,y
