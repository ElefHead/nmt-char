from unittest import TestCase
from nmt.datasets import Vocab
from nmt.datasets import batch_iter
from nmt.networks import CharEmbedding, Encoder
from torch import nn


sentences_words = [
    ['Human:', 'What', 'do', 'we', 'want?'],
    ['Computer:', 'Natural', 'language', 'processing!'],
    ['Human:', 'When', 'do', 'we', 'want', 'it?'],
    ['Computer:', 'When', 'do', 'we', 'want', 'what?']
]


class TestEncoder(TestCase):
    def test_encoder(self):
        vocab = Vocab.build(sentences_words, sentences_words)
        data = list(zip(sentences_words, sentences_words))
        data_generator = batch_iter(
            data=data,
            batch_size=4,
            shuffle=True
        )
        batch_src, _ = next(data_generator)
        source_length = [len(sent) for sent in batch_src]

        char_tensors = vocab.src.to_tensor(batch_src, tokens=False)
        token_tensors = vocab.src.to_tensor(batch_src, tokens=True)
        encoder = Encoder(input_size=300, hidden_size=1024, num_layers=2)
        embedding = nn.Embedding(
            num_embeddings=vocab.src.length(tokens=True),
            embedding_dim=300,
            padding_idx=vocab.src.pad_char_idx
        )
        token_embedding = embedding(token_tensors)
        c_embedding = CharEmbedding(
            num_embeddings=vocab.src.length(tokens=False),
            char_embedding_dim=50,
            embedding_dim=300,
            char_padding_idx=vocab.src.pad_char_idx
        )
        char_embedding = c_embedding(char_tensors)
        token_enc_hidden, (token_hidden, token_cell) = encoder(
            token_embedding,
            source_length
        )
        char_enc_hidden, (char_hidden, char_cell) = encoder(
            char_embedding,
            source_length
        )
        
        self.assertEqual(token_enc_hidden.size(), (4, 6, 2048))
        self.assertEqual(char_enc_hidden.size(), (4, 6, 2048))
        self.assertEqual(token_hidden.size(), (4, 1024))
        self.assertEqual(char_hidden.size(), (4, 1024))
        self.assertEqual(token_cell.size(), (4, 1024))
        self.assertEqual(char_cell.size(), (4, 1024))
