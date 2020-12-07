from unittest import TestCase
from nmt.networks import ModelEmbeddings
from nmt.datasets import Vocab

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
    def test_modelembedding(self):
        vocab = Vocab.build(sentences, sentences)
        char_tensor = vocab.src.to_tensor(sentences_words, tokens=False)
        embed_size = 1024
        char_embed_size = 50
        embedding = ModelEmbeddings(embed_size=embed_size,
                                    char_embed_size=char_embed_size,
                                    vocab_src=vocab.src)
        input_for_lstm = embedding(char_tensor)
        self.assertEqual(input_for_lstm.shape,
                         (*char_tensor.shape[:2], embed_size))
