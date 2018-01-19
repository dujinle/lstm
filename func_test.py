#!/usr/bin/python
#-*- coding:utf-8 -*-

import collections

counter = collections.Counter([1,2,3,4]);
print counter;
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]));
print count_pairs;
zipc = zip(*count_pairs);
print zipc;
words, _ = list(zipc);
print words;
zipc = zip(words,range(len(words)));
print zipc;
print dict(zipc)
