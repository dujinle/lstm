#!/usr/bin/python
#-*- coding:utf-8 -*-
#分词数据格式化 程序 对 数据 进行label标注
#例如： 我 们 的 朋 友
#       S  E  E  S  E
#S:start E:end B:between
import codecs,collections,sys
import copy, numpy as np
import djl_common as dc

sys.stdout = codecs.getwriter('utf8')(sys.stdout)

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
				wordlist.append(ii);
				if idx == 0:
					labelist.append('^');
				elif idx == len(i) - 1:
					labelist.append('#');
				else:
					labelist.append('@');
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
			fpw.write('\t'.join(ww) + '\n');
			fpw.write('\t'.join(wl) + '\n');
		else:
			ww = wordlist[idx:];
			wl = labelist[idx:];
			fpw.write('\t'.join(ww) + '\n');
			fpw.write('\t'.join(wl) + '\n');
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
				ww.append(word_to_id[wordlist[i]]);
				wl.append(word_to_id[labelist[i]]);
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		else:
			for i in range(idx,len(wordlist)):
				ww.append(word_to_id[wordlist[i]]);
				wl.append(word_to_id[labelist[i]]);
			fpw.write(' '.join(ww) + '\n');
			fpw.write(' '.join(wl) + '\n');
		idx = idx + 10;
	fpw.close();


if __name__ == '__main__':
	if len(sys.argv) <> 3:
		print 'Usage: %s vocab filename' %sys.argv[0];
		sys.exit(-1);

	vocab,id_to_word = dc._read_vocab(sys.argv[1]);
	read_data(sys.argv[2],vocab);

