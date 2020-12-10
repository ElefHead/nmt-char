import torch
from torch import nn


class CharDecoder(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int, num_layers: int,
                 output_size: int) -> None:
        """
        Initialize the character level decoder
        Check notebooks: char-decoding.ipynb for explanation
        @param input_size: input size for the lstm layer
            should be equal to character embedding dim.
        @param hidden_size: hidden units for lstm layer
        @param num_layers: number of layers for the lstm layer.
        @param output_size: number of output units on
            projection layer for score. Should be equal to
            vocab.tgt.length(tokens=False)
        """
        super(CharDecoder, self).__init__()
        self.char_decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self,
                x: torch.Tensor,
                dec_init: torch.Tensor) -> torch.Tensor:
        """
        Forward computation for char decoder
        @param x: character embedding tensor.
        @param dec_init: combined_output tensor from decoder network.

        @returns score: tensor of logits
        @returns dec_state: lstm hidden and cell states
        """
        dec_state = dec_init
        output, dec_state = self.char_decoder(x, dec_state)
        score = self.linear(output)
        return score, dec_state
