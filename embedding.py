import bz2
import random
import torch
from tqdm import tqdm
from icecream import ic


WORD_EMBEDDING_FILE = 'dataset/sgns.weibo.word.bz2'
UNK, PAD, BOS, EOS = '<unk> <pad> <bos> <eos>'.split()
SPECIAL_TOKEN_NUM = 4

token2embedding = {}

with bz2.open(WORD_EMBEDDING_FILE) as f:
    token_vectors = f.readlines()
    vob_size, dim = token_vectors[0].split()
print('load embedding file: {} end!'.format(WORD_EMBEDDING_FILE))

def get_embedding(vocabulary: set):
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