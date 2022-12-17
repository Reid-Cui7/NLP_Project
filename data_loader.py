import pandas as pd
import numpy as np
import jieba
from collections import defaultdict
import torch


def get_all_vocabulary(train_file_path, vocab_size):
    ...

def tokenizer(sentence, vocab: dict):
    UNK = 1
    ids = [vocab.get(word, UNK) for word in jieba.cut(sentence)]
    return ids


val_ratio = 0.2
train_file = 'dataset/train.csv'
content = pd.read_csv(train_file)
X_train, y_train = defaultdict(list), []
X_val, y_val = defaultdict(list), []
num_val = int(len(content) * val_ratio)

vocab2ids = None # TODO get embedding based on all vocabularies
for i, row in content.iterrows():
    label = row[1]
    sentence = row[-1]
    inputs = tokenizer(sentence, vocab2ids)

    if i < num_val:
        X_val['input_ids'].append(inputs)
        y_val.append(label)
    else:
        X_train['input_ids'].append(inputs)
        y_train.append(label)


label2id = {label: i for i, label in enumerate(np.unique(y_train))}
id2label = {i: label for label, i in label2id.items()}
y_train = torch.tensor([label2id[y] for y in y_train], dtype=torch.long)
y_val = torch.tensor([label2id[y] for y in y_val], dtype=torch.long)

print(X_train, y_train, X_val, y_val, label2id, id2label)