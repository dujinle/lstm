#!/usr/bin/python
#-*- coding:utf-8 -*-
#对训练数据和测试数据 进行过滤 生成字典文件 并去除 非法字符
import codecs,collections,sys
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

def read_data(filename):
	fp = codecs.open(filename,'r','utf8');
	wordlist = list();
	taglist = list();
	idx = 0;
	while True:
		line = fp.readline().strip('\r\n');
		if not line: break;
		for i in line:
			if idx % 2 <> 0:
				taglist.append(i);
			else:
				wordlist.append(i);
			idx = idx + 1;
	fp.close();
	return wordlist,taglist;

if __name__ == '__main__':
	if len(sys.argv) <> 2:
		print 'Usage: %s pretfile' %sys.argv[0];
		sys.exit(-1);
	w,t = read_data(sys.argv[1]);

	idx = 0;
	ws = ts = '';
	while True:
		if idx >= len(w): break;
		if t[idx] == '@':
			ws = ws + w[idx];
		elif t[idx] == '^':
			ws = ws + ' ' + w[idx];
		else:
			ws = ws + w[idx] + ' ';
		idx = idx + 1;
	print ws;
