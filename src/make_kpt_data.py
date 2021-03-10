import os
import json
import random
from collections import Counter
import pickle as pkl

import fairseq


def read_lines(filename):
    with open(filename) as file:
        lines = file.readlines()
    lines = [line.strip('\n') for line in lines]
    return lines

def extract_from_json(json_str, verbose=True):
    src = []
    tgt = []
    for idx in range(len(json_str)):
        if idx % 30000 == 0:
            if verbose:
                print('processing idx: ', idx)
        data = json.loads(json_str[idx])
        article = data['abstract']
        keyword = data['keyword']
        keyword = keyword.split(';')
        src.append(article)
        tgt.append(keyword)
    return src, tgt

def filter_absent_keyword(article_list, keyword_list):
    present_kw = []
    for article, keyword in zip(article_list, keyword_list):
        kw = []
        for word in keyword:
            if word in article:
                kw.append(word)
            elif word.lower() in article:
                kw.append(word.lower())
        present_kw.append(kw)
    return present_kw
    
    
    
train_lines = read_lines('data/KPTimes.train.jsonl')
test_lines = read_lines('data/KPTimes.valid.jsonl')
train_src, train_tgt = extract_from_json(train_lines)
test_src, test_tgt = extract_from_json(test_lines)
train_present_kw = filter_absent_keyword(train_src, train_tgt)
test_present_kw = filter_absent_keyword(test_src, test_tgt)


all_kw = sum(train_present_kw[::10], [])
kw_counter = Counter(all_kw)
kw_vocab = {}
for i in kw_counter.most_common(4000):
    kw, count = i
    kw_vocab[kw] = count
    
    
def mask_dataset(src, present_kw, vocab=kw_vocab):
    filter_src = []
    filter_tgt = []
    special_char = '&'
    for idx, (article, kw) in enumerate(zip(src, present_kw)):
    #     if idx % 50000 == 0:
    #         print(idx)
        if len(kw) > 0:
            tgt_arr = []
            for w in kw:
                if w in vocab or w.lower() in vocab:
                    article = article.replace(w, special_char)
                    article = article.replace(w.lower(), special_char)
                    tgt_arr.append(w.lower())
            if len(tgt_arr) > 0:
                filter_src.append(article)
                filter_tgt.append(tgt_arr)

    return filter_src, filter_tgt
    
    

train_filter_src, train_filter_tgt = mask_dataset(train_src, train_present_kw)
test_filter_src, test_filter_tgt = mask_dataset(test_src, test_present_kw)
def make_kw_line(kws):
#     kws = [kw.replace(' ', '-') for kw in kws]
    line = ' ; '.join(kws)
    return line

train_tgt_out = [make_kw_line(line) for line in train_filter_tgt]
test_tgt_out = [make_kw_line(line) for line in test_filter_tgt]


with open('data/mask/train.mask.src', 'w') as file:
    file.write('\n'.join(train_filter_src))
with open('data/mask/train.mask.tgt', 'w') as file:
    file.write('\n'.join(train_tgt_out))

with open('data/mask/test.mask.src', 'w') as file:
    file.write('\n'.join(test_filter_src))
with open('data/mask/test.mask.tgt', 'w') as file:
    file.write('\n'.join(test_tgt_out))