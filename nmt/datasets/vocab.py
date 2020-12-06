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

        # For handling tokens
        if token2id:  # restore from save
            self.token2id = token2id
            self.start_token = token2id["<s>"]
            self.end_token = token2id["</s>"]
            self.unk = token2id["<unk>"]
            self.pad = token2id["<pad>"]

            self.id2word = {v: k for k, v in token2id.items()}

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

            self.id2word = {}
            uniq_tokens = list(self.token2id.keys())
            token_freqs = count_corpus(tokens)
            uniq_tokens += [token for token, freq in token_freqs
                            if freq >= min_freq and token not in uniq_tokens]

            for token in uniq_tokens:
                self.token2id[token] = self.token2id.get(
                    token, len(self.token2id))
                self.id2word[self.token2id[token]] = token

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
        return len(self.token2id)

    def __getitem__(self, tokens: Union[List[str], Tuple[str], str]) -> \
            Union[List[int], int]:
        if not isinstance(tokens, (list, tuple)):
            return self.token2id.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def __contains__(self, token) -> bool:
        return token in self.token2id

    def __setitem__(self, key, value):
        raise ValueError("Vocabulary store is read only")

    def __repr__(self) -> str:
        return f"Vocab Store: Tokens [size={len(self)}], \
                Characters [size={len(self.char2id)}]"

    def to_tokens(self, indices: Union[List[int], Tuple[int], int]) -> \
            Union[List[int], int]:
        if not isinstance(indices, (list, tuple)):
            return self.id2word.get(indices, None)
        return [self.to_tokens(index) for index in indices]

    def len(self, tokens: bool = True) -> int:
        return len(self.token2id) if tokens else len(self.char2id)

    def sent2id(self, sents: List[List[str]]) -> List[List[int]]:
        return [self[sent] for sent in sents]

    def to_charid(self, char: Union[List[str], str]) -> int:
        if not isinstance(char, (list, tuple)):
            return self.char2id.get(char, self.unk)
        return [self.to_charid(c) for c in char]

    def word2char(self, tokens: Union[List[str], str]) -> \
            Union[List[List[int]], List[int]]:
        if not isinstance(tokens, (list, tuple)):
            return [self.char2id.get(char, self.unk)
                    for char in self.start_char + tokens + self.end_char]
        return [self.word2char(token) for token in tokens]

    def to_char(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        if not isinstance(indices, (list, tuple)):
            return self.id2char.get(indices, None)
        return [self.to_char(index) for index in indices]

    def sent2charid(self, sents: List[List[str]]) -> List[List[List[int]]]:
        return [self.word2char(sent) for sent in sents]

    def to_tensor(self, sents: List[List[str]],
                  tokens: bool,
                  device: torch.device) -> torch.Tensor:
        ids = self.sent2id(sents) if tokens else self.sent2charid(sents)
        pad_ids = pad_sents(ids, self[self.pad]) if tokens else pad_sents_char(
            ids, self.to_charid(self.pad))
        tensor_sents = torch.tensor(pad_ids, dtype=torch.long, device=device)
        return torch.t(tensor_sents) if tokens \
            else tensor_sents.permute([1, 0, 2])


class Vocab(object):
    def __init__(self, src_vocab: VocabStore = None,
                 tgt_vocab: VocabStore = None) -> None:
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents: List[List[str]], tgt_sents: List[List[str]],
              min_freq: int = 0) -> 'Vocab':
        assert len(src_sents) == len(tgt_sents)

        print("Initializing source vocab")
        src = VocabStore(src_sents, min_freq=min_freq)
        print(src)

        print("Initializing target vocab")
        tgt = VocabStore(tgt_sents, min_freq=min_freq)
        print(tgt)

        return Vocab(src, tgt)


class Vocab(object):
    def __init__(self, src_vocab: VocabStore = None,
                 tgt_vocab: VocabStore = None) -> None:
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents: List[List[str]], tgt_sents: List[List[str]],
              min_freq: int = 0) -> 'Vocab':
        assert len(src_sents) == len(tgt_sents)

        print("Initializing source vocab")
        src = VocabStore(src_sents, min_freq=min_freq)
        print(src)

        print("Initializing target vocab")
        tgt = VocabStore(tgt_sents, min_freq=min_freq)
        print(tgt)

        return Vocab(src, tgt)

    def save(self, filepath: Union[Path, str]):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True)

        with open(filepath, "w") as f:
            f.write(json.dumps(dict(src_token2id=self.src.token2id,
                    tgt_token2id=self.tgt.token2id)))

    @staticmethod
    def load(filepath: Union[Path, str]) -> Vocab:
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
        """ Representation of Vocab to be used
        when printing the object.
        """
        return f'Vocab src: {self.src}, tgt: {self.tgt}'
