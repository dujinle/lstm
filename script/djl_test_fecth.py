#!/usr/bin/python
#-*- coding:utf-8 -*-
#分词数据格式化 程序 对 数据 进行label标注
#例如： 我 们 的 朋 友
#       S  E  E  S  E
#S:start E:end B:between
import codecs,collections,sys
import copy, numpy as np
np.random.seed(0)
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

def read_vocab(filename):
	fp = codecs.open(filename,'r','utf8');
	word_to_id = dict();
	id_to_word = dict();
	while True:
		line = fp.readline().strip('\r\n');
		if not line: break;
		data = line.split('\t');
		word_to_id[data[0]] = data[1];
		id_to_word[data[1]] = data[0];
	return word_to_id,id_to_word;

def read_data(filename,word_to_id):
	fp = codecs.open(filename,'r','utf8');
	wordlist = list();
	labelist = list();
	while True:
		line = fp.readline().strip('\r\n');
		if not line: break;
		data = line.split();
		for i in data:
			idx = 0;
			for ii in i:
				if word_to_id.has_key(ii):
					wordlist.append(ii);
					if idx == 0:
						labelist.append('^');
					elif idx == len(i) - 1:
						labelist.append('#');
					else:
						labelist.append('@');
				else:
					print ii,'no found'
				idx = idx + 1;
	fp.close();

	outfile = filename.split('.');
	outname = filename;
	if len(outfile) > 1:
		outname = outfile[0] + '.lbl';
	else:
		outname = outname + '.lbl';

	fpw = codecs.open(outname,'w','utf8');
	idx = 0;
	while idx < len(wordlist):
		if idx + 10 < len(wordlist):
			ww = wordlist[idx:idx + 10];
			wl = labelist[idx:idx + 10];
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		else:
			ww = wordlist[idx:];
			wl = labelist[idx:];
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		idx = idx + 10;

	fpw.close();

	outname = outname + '.cors';
	fpw = codecs.open(outname,'w','utf8');
	idx = 0;
	while idx < len(wordlist):
		ww = list();
		wl = list();
		if idx + 10 < len(wordlist):
			for i in range(idx,idx + 10):
				ww.append(str(word_to_id[wordlist[i]]));
				wl.append(str(word_to_id[labelist[i]]));
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		else:
			for i in range(idx,len(wordlist)):
				ww.append(str(word_to_id[wordlist[i]]));
				wl.append(str(word_to_id[labelist[i]]));
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		idx = idx + 10;
	fpw.close();

if __name__ == '__main__':
	if len(sys.argv) <> 3:
		print 'Usage: %s vocab testfile';
		sys.exit(-1);

	word_to_id,id_to_word = read_vocab(sys.argv[1]);
	read_data(sys.argv[2],word_to_id);

