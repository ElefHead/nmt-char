import torch
from torch import nn
from torch.nn.functional import softmax

from nmt.networks import CharEmbedding

from typing import Tuple


def attention(value: torch.Tensor,
              key: torch.Tensor,
              query: torch.Tensor,
              enc_masks: torch.Tensor = None) -> torch.Tensor:

    query_unsqueezed = query.unsqueeze(dim=2)
    score = key.bmm(query_unsqueezed)
    score = score.squeeze(dim=2)

    if enc_masks is not None:
        score.data.masked_fill_(
            enc_masks.byte(),
            -float('inf')
        )

    attention_weights = softmax(score, dim=1)
    attention_weights = attention_weights.unsqueeze(dim=1)

    context_vector = attention_weights.bmm(value)
    context_vector = context_vector.squeeze(dim=1)

    return attention_weights, context_vector


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
                enc_projection: torch.Tensor,
                o_prev: torch.Tensor,
                enc_masks: torch.Tensor = None) -> torch.Tensor:

        dec_state = dec_init_state
        x = self.embedding(x)

        Ybar_t = torch.cat([x.squeeze(dim=0), o_prev], dim=1)

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state

        attention_scores, context_vector = attention(
            enc_hidden, enc_projection, dec_hidden, enc_masks)

        U_t = torch.cat([context_vector, dec_hidden], dim=1)
        V_t = self.combined_projection(U_t)
        output = self.dropout(V_t.tanh())

        return output, dec_state, attention_scores
