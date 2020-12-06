from collections import Counter
from typing import Union, List
from functools import reduce
from pathlib import Path
import numpy as np


def count_corpus(tokens: Union[List[str], List[List[str]]],
                 sort: bool = True) -> List[tuple]:
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    counts = Counter(tokens).items()
    if sort:
        counts = sorted(counts, key=lambda x: x[0])
        counts = sorted(counts, key=lambda x: x[1], reverse=True)
    return counts


def pad_sents_char(sents: List[List[List[int]]],
                   pad_token: int,
                   max_word_length: int = 21) -> List[List[List[int]]]:
    max_sent_length = reduce(max, map(len, sents))
    empty_word = [pad_token] * max_word_length
    sents_padded = []
    for sent in sents:
        sent_padded = [(word + [pad_token]*(max_word_length - len(word)))
                       [:max_word_length] for word in sent]
        sents_padded.append(
            sent_padded + [empty_word] * (max_sent_length - len(sent)))
    return sents_padded


def pad_sents(sents: List[List[int]], pad_token: int) -> List[List[int]]:
    sents_padded = []

    max_sent_length = reduce(max, map(len, sents))
    for sent in sents:
        sents_padded.append(sent + [pad_token]*(max_sent_length - len(sent)))

    return sents_padded


def read_corpus(filepath: Union[str, Path],
                is_target: bool = False) -> List[str]:
    """ Read file, where each sentence is dilineated by a `\n`.
    @param filepath (str, Path): path to file containing corpus
    @param is_target (bool): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            sent = line.strip().split()
            # only append <s> and </s> to the target sentence
            if is_target:
                sent = ['<s>'] + sent + ['</s>']
            data.append(sent)

    return data


def batch_iter(data: Union[tuple, list],
               batch_size: int,
               shuffle: bool = False) -> tuple:
    """
    Yield batches of source and target sentences reverse sorted by length
    (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing
        source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = int(np.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
