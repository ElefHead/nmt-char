from unittest import TestCase
from nmt.datasets import Vocab
from nmt.datasets import batch_iter

sentences_words = [
    ['Human:', 'What', 'do', 'we', 'want?'],
    ['Computer:', 'Natural', 'language', 'processing!'],
    ['Human:', 'When', 'do', 'we', 'want', 'it?'],
    ['Computer:', 'When', 'do', 'we', 'want', 'what?']
]

sentences = [
    "Human: What do we want?",
    "Computer: Natural language, processing!",
    "Human: When do we want it?",
    "Computer: When do we want what?"
]


class TestVocab(TestCase):
    def test_build(self):
        vocab = Vocab.build(sentences, sentences)
        self.assertEqual(len(vocab.src), 17)
        self.assertEqual(len(vocab.tgt), 17)

        vocab1 = Vocab.build(sentences_words, sentences)
        self.assertEqual(len(vocab1.src), 17)
        self.assertEqual(len(vocab1.tgt), 17)

    def test_lengths(self):
        vocab = Vocab.build(sentences, sentences_words)
        self.assertIsInstance(len(vocab.src), int)
        self.assertIsInstance(vocab.src.length(tokens=True), int)
        self.assertIsInstance(vocab.src.length(tokens=False), int)

    def test_totensor(self):
        vocab = Vocab.build(sentences, sentences_words)
        token_tensors = vocab.src.to_tensor(sentences_words, tokens=True)
        char_tensors = vocab.src.to_tensor(sentences_words, tokens=False)
        self.assertEqual(token_tensors.shape, (6, 4))
        self.assertEqual(char_tensors.shape, (6, 4, 21))

    def test_batchiter(self):
        data = list(zip(sentences_words, sentences_words))
        for src, tgt in batch_iter(data,
                                   batch_size=2, shuffle=True):
            self.assertEqual(len(src), len(tgt))
