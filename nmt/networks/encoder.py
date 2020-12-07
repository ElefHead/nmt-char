import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import List


class Encoder(nn.Module):
    def __init__(self, input_size: torch.Tensor,
                 hidden_size: int, num_layers: int) -> None:
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            bidirectional=True
        )
        self.hidden_projection = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False
        )
        self.cell_projection = nn.Linear(
            in_features=hidden_size * 2,
            out_features=hidden_size,
            bias=False
        )

    def forward(self, x: torch.Tensor, source_lengths: List[int]):
        # x is token embeddings

        x = pack_padded_sequence(x, lengths=source_lengths)
        enc_output, (last_hidden, last_cell) = self.encoder(x)

        enc_output, _ = pad_packed_sequence(enc_output)
        enc_output = enc_output.permute([1, 0, 2])

        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        init_decoder_hidden = self.hidden_projection(last_hidden)

        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)
        init_decoder_cell = self.cell_projection(last_cell)

        return enc_output, (init_decoder_hidden, init_decoder_cell)
