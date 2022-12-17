import pandas as pd
import numpy as np
import jieba
from collections import defaultdict
import torch
from operator import add
from functools import reduce
from collections import Counter

def add_with_print(all_corpus):
    # 程序某一步骤运行很慢时, 可以在那个步骤上加入一个装饰器, 让你知道这一步运行到哪里了, 到底有没有运行
    add_with_print.i = 0

    def _warp(a, b):
        add_with_print.i += 1
        print('{}/{}'.format(add_with_print.i, len(all_corpus)), end=' ')
        return a + b
    
    return _warp

def get_all_vocabulary(train_file_path, vocab_size):
    CUT, SENTENCE = 'cur', 'sentence'
    corpus = pd.read_csv(train_file_path)
    corpus[CUT] = corpus[SENTENCE].apply(lambda s: ' '.join(list(jieba.cut(s))))
    sentence_counters = map(Counter, map(lambda s: s.split(), corpus[CUT].values))
    chose_words = reduce(add_with_print(corpus), sentence_counters).most_common(vocab_size)

    return [w for w, _ in chose_words]

def tokenizer(sentence, vocab: dict):
    UNK = 1
    ids = [vocab.get(word, UNK) for word in jieba.cut(sentence)]

    return ids

def get_train_data(train_file, vocab2ids):
    val_ratio = 0.2
    content = pd.read_csv(train_file)
    num_val = int(len(content) * val_ratio)
    
    LABEL, SENTENCE = 'label', 'sentence'

    labels = content[LABEL].values
    content['input_ids'] = content[SENTENCE].apply(lambda s: ' '.join([str(id_) for id_ in tokenizer(s, vocab2ids)]))
    sentence_ids = np.array([[int(id_) for id_ in v.split()] for v in content['input_ids'].values])

    # shuffle ids
    ids = np.random.choice(range(len(content)), size=len(content))
    
    train_ids = ids[num_val:]
    val_ids = ids[:num_val]
    X_train, y_train = sentence_ids[train_ids], labels[train_ids]
    X_val, y_val = sentence_ids[val_ids], labels[val_ids]

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