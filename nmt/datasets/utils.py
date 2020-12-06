from collections import Counter
from typing import Union, List
from functools import reduce


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
