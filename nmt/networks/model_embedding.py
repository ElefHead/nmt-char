from torch import nn
from nmt.layers import Highway, CharCNNEmbedding
from nmt.datasets import VocabStore


class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size: int, char_embed_size: int,
                 vocab_src: VocabStore, cnn_kernel_size: int = 5,
                 dropout_prob: float = 0.3) -> None:
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.char_embed_size = char_embed_size

        pad_char_id = vocab_src.pad_char_idx

        self.char_embed = nn.Embedding(
            num_embeddings=vocab_src.length(tokens=False),
            embedding_dim=char_embed_size,
            padding_idx=pad_char_id
        )
        self.cnn_embed = CharCNNEmbedding(
            in_channels=char_embed_size,
            out_channels=embed_size,
            kernel_size=cnn_kernel_size
        )
        self.highway = Highway(in_features=embed_size, out_features=embed_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        sentence_length, batch_size, word_length = x.size()

        embed_t = self.char_embed(x)
        embed_t = embed_t.view(-1, word_length,
                               self.char_embed_size).permute([0, 2, 1])

        cnn_embed_t = self.cnn_embed(embed_t)
        cnn_embed_t = cnn_embed_t.squeeze(dim=2)

        highway_t = self.highway(cnn_embed_t)

        out = self.dropout(highway_t)

        out = out.view(sentence_length, batch_size, self.embed_size)
        return out
