import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

def get_examples(file):
    with open(file) as f:
        examples = []
        for example in f.readlines():
            examples.append(example.strip())
    return examples
            

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

train_nl = get_examples("data/train.nl")
dev_nl = get_examples("data/dev.nl")

train_sql = get_examples("data/train.sql")
dev_sql = get_examples("data/dev.sql")

train_nl = ["translate english to sql: " + ex for ex in train_nl]
dev_nl = ["translate english to sql: " + ex for ex in dev_nl]
train_sql = [ex.replace("AND 1 = 1", "") for ex in train_sql]
dev_sql = [ex.replace("AND 1 = 1", "") for ex in dev_sql]

train_nl = [tokenizer.encode(ex) for ex in train_nl]
dev_nl = [tokenizer.encode(ex) for ex in dev_nl]

train_sql = [tokenizer.encode(ex) for ex in train_sql]
dev_sql = [tokenizer.encode(ex) for ex in dev_sql]

train_nl_vocab = []
train_nl_avg_len = 0

dev_nl_vocab = []
dev_nl_avg_len = 0

train_sql_vocab = []
train_sql_avg_len = 0
dev_sql_vocab = []
dev_sql_avg_len = 0

print("Num Train Examples: ", len(train_nl))
print("Num dev Examples: ", len(dev_nl))


for ex in train_nl:
    train_nl_avg_len += len(ex)
    for token in ex:
        train_nl_vocab.append(token)

train_nl_vocab = set(train_nl_vocab)


for ex in dev_nl:
    dev_nl_avg_len += len(ex)
    for token in ex:
        dev_nl_vocab.append(token)

dev_nl_vocab = set(dev_nl_vocab)


for ex in train_sql:
    train_sql_avg_len += len(ex)
    for token in ex:
        train_sql_vocab.append(token)

train_sql_vocab = set(train_sql_vocab)


for ex in dev_sql:
    dev_sql_avg_len += len(ex)
    for token in ex:
        dev_sql_vocab.append(token)

dev_sql_vocab = set(dev_sql_vocab)

print("train_nl_avg_len: ", train_nl_avg_len/len(train_nl))
print("dev_nl_avg_len: ", dev_nl_avg_len/len(dev_nl))
print("train_sql_avg_len: ", train_sql_avg_len/len(train_sql))
print("dev_sql_avg_len: ", dev_sql_avg_len/len(dev_sql))

print("train_nl_vocab: ", len(train_nl_vocab))
print("dev_nl_vocab: ", len(dev_nl_vocab))
print("train_sql_vocab: ", len(train_sql_vocab))
print("dev_sql_vocab: ", len(dev_sql_vocab))
























        