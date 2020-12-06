from typing import List, Dict, Union, Tuple
import torch
from .utils import pad_sents, pad_sents_char, count_corpus
from pathlib import Path
import json


class VocabStore(object):
    """
    Will store source or target vocabulary
    """

    def __init__(self, tokens: List[List[str]] = None,
                 token2id: Dict[str, int] = None,
                 min_freq: int = 0,
                 reserved_tokens: Dict[str, str] = None) -> None:
        """
        Constructs the vocabulary

        @param tokens: List of tokenized sentences
        @param token2id: Dictionary of token to id mapping when loading
            from saved json vocabstore.
        @param min_freq: Int(default = 0) discard threshold for rare words
        @param reserved_tokens: Dict of start, end, pad, unk tokens. Key is
            the name and value is the token. Eg. {'start': '<s>'}
        """
        # For handling tokens
        if token2id:  # restore from save
            self.token2id = token2id
            self.start_token = token2id["<s>"]
            self.end_token = token2id["</s>"]
            self.unk = token2id["<unk>"]
            self.pad = token2id["<pad>"]

            self.id2token = {v: k for k, v in token2id.items()}

        else:  # build new
            self.token2id = {}
            if not reserved_tokens:
                reserved_tokens = {}

            reserved_tokens["unk"] = reserved_tokens.get("unk", "<unk>")
            reserved_tokens["pad"] = reserved_tokens.get("pad", "<pad>")
            reserved_tokens["start"] = reserved_tokens.get("start", "<s>")
            reserved_tokens["end"] = reserved_tokens.get("end", "</s>")

            self.start_token, self.token2id[reserved_tokens['start']
                                            ] = reserved_tokens["start"], 1
            self.end_token, self.token2id[reserved_tokens['end']
                                          ] = reserved_tokens["end"], 2
            self.unk, self.token2id[reserved_tokens['unk']
                                    ] = reserved_tokens["unk"], 3
            self.pad, self.token2id[reserved_tokens['pad']
                                    ] = reserved_tokens["pad"], 0

            if not tokens:
                tokens = []

            self.id2token = {}
            uniq_tokens = list(self.token2id.keys())
            token_freqs = count_corpus(tokens)
            uniq_tokens += [token for token, freq in token_freqs
                            if freq >= min_freq and token not in uniq_tokens]

            for token in uniq_tokens:
                self.token2id[token] = self.token2id.get(
                    token, len(self.token2id))
                self.id2token[self.token2id[token]] = token

        # For handling chars

        self.char_list = list(
            """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;
            .!?:'\"/\\|_@#$%^&*~`+-=<>()[]""".replace("\n", "")
        )
        self.char2id = {}
        self.char2id[self.pad] = 0
        self.start_char, self.char2id["{"] = "{", 1
        self.end_char, self.char2id["}"] = "}", 2
        self.char2id[self.unk] = 3

        for c in self.char_list:
            self.char2id[c] = self.char2id.get(c, len(self.char2id))

        self.id2char = {v: k for k, v in self.char2id.items()}

    def __len__(self) -> int:
        """
        Overrides len() function.
        @returns int: number of tokens in vocabulary
        """
        return len(self.token2id)

    def __getitem__(self, tokens: Union[List[str], Tuple[str], str]) -> \
            Union[List[int], int]:
        """
        Retrieves token's index. Can handle a list of tokens
            or just a single token.
        @param tokens: List, tuple of tokens or single token
        @returns list or int: if input was a container then a list of indices
            otherwise a single index. Unk index if token not in vocab.
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token2id.get(tokens, self.token2id[self.unk])
        return [self.__getitem__(token) for token in tokens]

    def __contains__(self, token) -> bool:
        """
        Checks if token is in vocabulary.
        @param token (str): token to look up.
        @returns bool: True if token in vocab else False
        """
        return token in self.token2id

    def __setitem__(self, key, value):
        """
        Raises error; Vocab is read only
        """
        raise ValueError("Vocabulary store is read only")

    def __repr__(self) -> str:
        """
        Sets print representation
        """
        return f"""Vocab Store: Tokens [size={len(self)}],
                Characters [size={len(self.char2id)}]""".replace("\n", " ")

    def to_tokens(self, indices: Union[List[int], Tuple[int], int]) -> \
            Union[List[str], str]:
        """
        Converts indices to tokens
        @param indices: List of indices or a single index
        @returns list of corresponding tokens if indices is a list
            else a single token
        """
        if not isinstance(indices, (list, tuple)):
            return self.id2token.get(indices, None)
        return [self.to_tokens(index) for index in indices]

    def length(self, tokens: bool = True) -> int:
        """
        Another function for computing length of vocab for handling
        tokens as well as characters.
        @param tokens: bool (default = True) if set to true
            it will return same as len(vocab) - number of tokens in vocab
            if set to false, it will return number of characters in vocab
        @retuns int: number of tokens or number of characters in vocab
        """
        return len(self.token2id) if tokens else len(self.char2id)

    def add(self, token: str) -> int:
        """
        Add token to vocab if previouslu unseen.
        @param token: token string to be added
        @returns index of given token
        """
        self.token2id[token] = self.token2id.get(token, len(self))
        idx = self.token2id[token]
        self.id2token[idx] = token
        return idx

    def sent2id(self, sents: List[List[str]]) -> List[List[int]]:
        """
        Converts a sentence (list of list of token string) to
        corresponding indices (list of list of index int)
        @params: sents : input sentences as list of list of token strings
        @returns: corresponding indices (list of list of ints)
        """
        return [self[sent] for sent in sents]

    def to_charid(self, char: Union[List[str], str]) -> Union[List[int], int]:
        """
        Converts chars to corresponding char indices
        @params char: one character or list of characters
        @returns list of corresponding char indices if input is list
            else single char index
        """
        if not isinstance(char, (list, tuple)):
            return self.char2id.get(char, self.char2id[self.unk])
        return [self.to_charid(c) for c in char]

    def word2char(self, tokens: Union[List[str], str]) -> \
            Union[List[List[int]], List[int]]:
        """
        Converts token(s) to corresponding char indices
        @param tokens: single token or a list of tokens
        @returns list of indices with start of word and
            end of word char indices appended.
        """
        if not isinstance(tokens, (list, tuple)):
            return [self.char2id.get(char, self.unk)
                    for char in self.start_char + tokens + self.end_char]
        return [self.word2char(token) for token in tokens]

    def to_char(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Converts indices to corresponding chars.
        @param indices: single index or list of indices
        @returns single char or a list of chars.
        """
        if not isinstance(indices, (list, tuple)):
            return self.id2char.get(indices, None)
        return [self.to_char(index) for index in indices]

    def sent2charid(self, sents: List[List[str]]) -> List[List[List[int]]]:
        """
        Converts sentences to corresponding char indices
        @param sents: list of list of token strings
        @returns indices: list of list of list of char indices
        """
        return [self.word2char(sent) for sent in sents]

    def to_tensor(self, sents: List[List[str]],
                  tokens: bool,
                  device: torch.device = 'cpu',
                  max_word_length: int = 21) -> torch.Tensor:
        """
        Converts sentences to token or char index tensors
        @param sents: list of list of token strings.
        @param tokens: bool representing requirement of token
            tensor or character tensor. Set true for token tensors.
            Set false for character tensors.
        @param device (default: "cpu"): cpu or gpu for loading the tensor.
        @max_word_length (default: 21) : maximum allowed length of words.
            any longer word will be truncated. shorter words will get padded.

        @returns token tensor : torch.Tensor of
            shape (sentence_length, batch_size) if tokens true
            else char tensor: torch.Tensor of
            shape (sentence_length, batch_size, max_word_length)
        """
        ids = self.sent2id(sents) if tokens else self.sent2charid(sents)
        pad_ids = pad_sents(ids, self[self.pad]) if tokens else \
            pad_sents_char(ids, self.to_charid(self.pad),
                           max_word_length=max_word_length)
        tensor_sents = torch.tensor(pad_ids, dtype=torch.long, device=device)
        return torch.t(tensor_sents) if tokens \
            else tensor_sents.permute([1, 0, 2])


class Vocab(object):
    def __init__(self, src_vocab: VocabStore = None,
                 tgt_vocab: VocabStore = None) -> None:
        """
        Create vocabulary for NMT task.

        @param src_vocab (VocabStore): VocabStore for source language
        @param tgt_vocab (VocabStore): VocabStore for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents: Union[List[str], List[List[str]]],
              tgt_sents: Union[List[str], List[List[str]]],
              min_freq: int = 0) -> 'Vocab':
        """
        Build Vocabulary for NMT task.

        @param src_sents: Source sentences
        @param tgt_sents: Target sentences
        @param min_freq (int): if token occurs n < freq_cutoff times, drop it.

        @returns vocab object containing source and target vocabulary.
        """
        if src_sents and isinstance(src_sents[0], str):
            src_sents = [line.split() for line in src_sents]

        if tgt_sents and isinstance(tgt_sents[0], str):
            tgt_sents = [line.split() for line in tgt_sents]

        assert len(src_sents) == len(tgt_sents)

        print("Initializing source vocab")
        src = VocabStore(src_sents, min_freq=min_freq)
        print(src)

        print("Initializing target vocab")
        tgt = VocabStore(tgt_sents, min_freq=min_freq)
        print(tgt)

        return Vocab(src, tgt)

    def save(self, filepath: Union[Path, str]):
        """
        Save Vocab to file as JSON dump.
        @param filepath (str, Path): file path to vocab file
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True)

        with open(filepath, "w") as f:
            f.write(json.dumps(dict(src_token2id=self.src.token2id,
                                    tgt_token2id=self.tgt.token2id)))

    @staticmethod
    def load(filepath: Union[Path, str]) -> 'Vocab':
        """
        Load vocabulary from JSON dump.
        @param filepath (str, Path): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        vocab_data = {}
        with open(filepath, "r") as f:
            vocab_data = json.loads(f.read())

        src_token2id = vocab_data["src_token2id"]
        tgt_token2id = vocab_data["tgt_token2id"]

        return Vocab(
            VocabStore(token2id=src_token2id),
            VocabStore(token2id=tgt_token2id)
        )

    def __repr__(self):
        """
        Representation of Vocab to be used
        when printing the object.
        """
        return f'Vocab src: {self.src}, tgt: {self.tgt}'
