import bz2
import random
import torch
from tqdm import tqdm
from icecream import ic
import torch.nn as nn
import torch.nn.functional as F


WORD_EMBEDDING_FILE = 'dataset/sgns.weibo.word.bz2'
UNK, PAD, BOS, EOS = '<unk> <pad> <bos> <eos>'.split()
SPECIAL_TOKEN_NUM = 4

token2embedding = {}

def get_embedding(vocabulary: set):
    with bz2.open(WORD_EMBEDDING_FILE) as f:
        token_vectors = f.readlines()
        vob_size, dim = token_vectors[0].split()

        for line in tqdm(token_vectors[1:]):
            tokens = line.split()
            token = tokens[0].decode("utf-8")
            if token in vocabulary:
                token2embedding[token] = list(map(float, tokens[1:]))
                assert len(token2embedding[token]) == int(dim)

    token2id = {token: _id for _id, token in enumerate(token2embedding.keys(), SPECIAL_TOKEN_NUM)}
    token2id[PAD] = 0
    token2id[UNK] = 1
    token2id[BOS] = 2
    token2id[EOS] = 3

    id2vec = {token2id[token]: embedding for token, embedding in token2embedding.items()}
    id2vec[0] = [0.] * int(dim)
    id2vec[1] = [0.] * int(dim)
    id2vec[2] = [random.uniform(-1, 1)] * int(dim)
    id2vec[3] = [random.uniform(-1, 1)] * int(dim)

    embedding = [id2vec[_id] for _id in range(len(id2vec))]

    return torch.tensor(embedding, dtype=torch.float), token2id, len(vocabulary) + SPECIAL_TOKEN_NUM


class TextCNN(nn.Module):
    def __init__(self, word_embedding, each_filter_num, filter_heights, drop_out, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_embedding, freeze=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=each_filter_num,
                      kernel_size=(h, word_embedding.shape[0]))
            for h in filter_heights
        ])

        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(each_filter_num * len(filter_heights), num_classes)

    def conv_and_pool(self, x, conv):
            x = F.relu(conv(x)).squeeze(3)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)

            return x

    def forward(self, input_ids=None, labels=None):
        word_embeddings = self.embedding(input_ids)
        sentence_embeddings = word_embeddings.unsqueeze(1)

        out = torch.cat([self.conv_and_pool(sentence_embeddings, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        if labels:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, labels)
            outputs = (loss, ) + outputs

        return outputs



if __name__ == '__main__':
    some_test_words = ['法国', '三', '连冠', ',', '格祖', '加油']

    embedding, token2id, _ = get_embedding(set(some_test_words))

    ic(embedding[token2id['法国']])
    ic(embedding[token2id['加油']])