import os
import json
from collections import Counter

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'checkpoints/wiki/',
    checkpoint_file='checkpoint1.pt',
    data_name_or_path='wiki/bin'
)

bart.cuda()
bart.eval()
bart.half()


def read_lines(filename):
    with open(filename) as file:
        lines = file.readlines()
    lines = [line.strip('\n') for line in lines]
    return lines

def inference_batch(batch, min_len=10):
    assert type(batch) == list
    assert type(batch[0]) == str
    with torch.no_grad():
        pred = bart.sample(batch, beam=1, lenpen=2.0, max_len_b=140, min_len=min_len, no_repeat_ngram_size=3)
    return pred

def process_json(json_lines):
    article = []
    keyword = []
    for line in json_lines:
        data = json.loads(line)
        article.append(data['abstract'])
        keyword.append(data['keyword'].split(';'))
    return article, keyword

raw_json = read_lines('data/KPTimes.test.jsonl')
raw_article, raw_keyword = process_json(raw_json)


batch_sz = 8
pred = []
for idx in tqdm(range(0, min(len(raw_article), 5000), batch_sz)):
    out = inference_batch(raw_article[idx : idx+batch_sz])
    pred += out
    
stem = PorterStemmer().stem
# tokenize, stem, join, lower
tsjl = lambda s: ' '.join(map(stem, word_tokenize(s))).lower()

# tsjl('interesting')

def compute_prf(cand, ref):
    cand_set = set(cand)
    ref_set = set(ref)
    p = 0
    for word in cand_set:
        if word in ref_set:
            p += 1
    r = 0
    for word in ref_set:
        if word in cand_set:
            r += 1
    p = p / len(cand_set)
    r = r / len(ref_set)
    f = (p*r) / (p+r+1e-5) * 2
    return p, r, f

def compute_ap(cand, ref):
    cand_arr = []
    for word in cand:
        if word not in cand_arr:
            cand_arr.append(word)
    ref_set = set(ref)
    cumulate = 0
    ap = []
    for idx, word in enumerate(cand_arr):
        if word in ref_set:
            cumulate += 1
            ap.append(cumulate / (idx+1))
        else:
            ap.append(0)
    return sum(ap) / len(ap)

precisions = []
recalls = []
f_scores = []
average_p = []

for cand, ref in tqdm(zip(pred, raw_keyword)):
    cand = cand.split(' ; ')
    cand = [tsjl(w) for w in cand]
    ref = [tsjl(w) for w in ref]
    max_len = min(len(ref), 10)
    p, r, f = compute_prf(cand, ref[:max_len])
    ap = compute_ap(cand, ref[:max_len])

    precisions.append(p)
    recalls.append(r)
    f_scores.append(f)
    average_p.append(ap)

P = sum(precisions) / len(precisions) * 100.0
R = sum(recalls) / len(recalls) * 100.0
F = sum(f_scores) / len(f_scores) * 100.0
MAP = sum(average_p) / len(average_p) * 100.0

print(P, R, F, MAP)