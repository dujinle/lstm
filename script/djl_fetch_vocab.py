#!/usr/bin/python
#-*- coding:utf-8 -*-
#对训练数据和测试数据 进行过滤 生成字典文件 并去除 非法字符
import codecs,collections,sys
import copy, numpy as np
np.random.seed(0)
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

def read_data(filename):
	fp = codecs.open(filename,'r','utf8');
	wordlist = list();
	while True:
		line = fp.readline().strip('\r\n');
		if not line: break;
		data = line.split();
		for i in data:
			for ii in i:
				wordlist.append(ii);
	fp.close();
	return wordlist;

if __name__ == '__main__':
	if len(sys.argv) <> 3:
		print 'Usage: %s train_data test_data' %sys.argv[0];
		sys.exit(-1);

	vocab_list = list();
	vocab_list.extend(read_data(sys.argv[1]));
	vocab_list.extend(read_data(sys.argv[2]));
	counter = collections.Counter(vocab_list);
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]));
	words, _ = list(zip(*count_pairs));
	word_to_id = dict(zip(words, range(len(words))));

	vocab = dict();
	id_to_word = dict();
	for key in word_to_id.keys():
		id_to_word[word_to_id[key]] = key;

	for wid in id_to_word.keys():
		print id_to_word[wid] + '\t' + str(wid);
