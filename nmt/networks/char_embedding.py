from torch import nn
from nmt.layers import Highway, CharCNNEmbedding


class CharEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, char_embedding_dim: int,
                 embedding_dim: int, char_padding_idx: int,
                 cnn_kernel_size: int = 5,
                 dropout_prob: float = 0.3) -> None:
        super(CharEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.char_embedding_dim = char_embedding_dim

        self.char_embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=char_embedding_dim,
            padding_idx=char_padding_idx
        )
        self.cnn_embed = CharCNNEmbedding(
            in_channels=char_embedding_dim,
            out_channels=embedding_dim,
            kernel_size=cnn_kernel_size
        )
        self.highway = Highway(
            in_features=embedding_dim,
            out_features=embedding_dim
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        sentence_length, batch_size, word_length = x.size()

        embed_t = self.char_embed(x)
        embed_t = embed_t.view(-1, word_length,
                               self.char_embedding_dim).permute([0, 2, 1])

        cnn_embed_t = self.cnn_embed(embed_t)

        highway_t = self.highway(cnn_embed_t)

        out = self.dropout(highway_t)

        out = out.view(sentence_length, batch_size, self.embedding_dim)
        return out
