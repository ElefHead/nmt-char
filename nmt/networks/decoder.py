import torch
from torch import nn

from nmt.layers import Attention
from nmt.networks import CharEmbedding

from typing import Tuple


class Decoder(nn.Module):
    def __init__(self, num_embeddings: int,
                 embedding_dim: int, hidden_size: int,
                 char_padding_idx: int, char_embedding_dim: int = 50,
                 bias: bool = True, dropout_prob: float = 0.3):
        super(Decoder, self).__init__()
        self.embedding = CharEmbedding(
            num_embeddings=num_embeddings,
            char_embedding_dim=char_embedding_dim,
            embedding_dim=embedding_dim,
            char_padding_idx=char_padding_idx
        )
        self.attention = Attention(
            in_features=hidden_size*2,
            out_features=hidden_size
        )
        self.decoder = nn.LSTMCell(
            input_size=embedding_dim + hidden_size,
            hidden_size=hidden_size,
            bias=bias
        )
        self.combined_projection = nn.Linear(
            in_features=hidden_size*3,
            out_features=hidden_size,
            bias=False
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor,
                enc_hidden: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor],
                o_prev: torch.Tensor,
                enc_masks: torch.Tensor = None) -> torch.Tensor:

        dec_state = dec_init_state
        x = self.embedding(x)

        Ybar_t = torch.cat([x.squeeze(dim=0), o_prev], dim=1)

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state

        attention_scores, context_vector = self.attention(
            enc_hidden, dec_hidden, enc_masks)

        U_t = torch.cat([context_vector, dec_hidden], dim=1)
        V_t = self.combined_projection(U_t)
        output = self.dropout(V_t.tanh())

        return output, dec_state, attention_scores
