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

