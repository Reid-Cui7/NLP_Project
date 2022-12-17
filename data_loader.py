import pandas as pd
import numpy as np
import jieba
from collections import defaultdict
import torch
from operator import add
from functools import reduce
from collections import Counter


def get_all_vocabulary(train_file_path, vocab_size):
    CUT, SENTENCE = 'cur', 'sentence'
    corpus = pd.read_csv(train_file_path)
    corpus[CUT] = corpus[SENTENCE].apply(lambda s: ' '.join(list(jieba.cut(s))))
    sentence_counters = map(Counter, map(lambda s: s.split(), corpus[CUT].values))
    chose_words = reduce(add, sentence_counters).most_common(vocab_size)

    return [w for w, _ in chose_words]

def tokenizer(sentence, vocab: dict):
    UNK = 1
    ids = [vocab.get(word, UNK) for word in jieba.cut(sentence)]

    return ids

def get_train_data(train_file):
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

    return X_train, y_train, X_val, y_val, label2id, id2label


if __name__ == "__main__":
    vocab_size = 10000
    vocabulary = get_all_vocabulary(train_file_path='dataset/train.csv', vocab_size=vocab_size)
    assert isinstance(vocabulary, list)
    assert isinstance(vocabulary[0], str)
    assert len(vocabulary) <= vocab_size