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
        self.attention = Attention()
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

    def forward(self, output: torch.Tensor,
                enc_hidden: torch.Tensor,
                enc_projection: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor],
                enc_masks: torch.Tensor = None,
                device: torch.device = 'cpu') -> torch.Tensor:

        dec_state = dec_init_state
        batch_size, sent_length, _ = enc_hidden.size()

        o_prev = torch.zeros(batch_size, self.hidden_size, device=device)
        combined_outputs = []

        for Y_t in torch.split(output, 1, dim=0):
            Ybar_t = torch.cat([Y_t.squeeze(dim=0), o_prev], dim=1)

            dec_state = self.decoder(Ybar_t, dec_state)
            dec_hidden, dec_cell = dec_state

            a_t = self.attention(
                enc_hidden, enc_projection, dec_hidden, enc_masks)

            U_t = torch.cat([a_t, dec_hidden], dim=1)
            V_t = self.combined_projection(U_t)
            o_t = self.dropout(V_t.tanh())

            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs
