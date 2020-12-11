import torch
from torch import nn

from nmt.layers import Attention

from typing import Tuple


class Decoder(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 dropout_prob: float = 0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(
            in_features=hidden_size*2,
            out_features=hidden_size
        )
        self.decoder = nn.LSTMCell(
            input_size=input_size,
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

        Ybar_t = torch.cat([x.squeeze(dim=0), o_prev], dim=1)

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state

        attention_scores, context_vector = self.attention(
            enc_hidden, dec_hidden, enc_masks
        )

        U_t = torch.cat([context_vector, dec_hidden], dim=1)
        V_t = self.combined_projection(U_t)
        output = self.dropout(V_t.tanh())

        return output, dec_state, attention_scores
