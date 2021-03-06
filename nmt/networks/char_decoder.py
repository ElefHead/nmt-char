import torch
from torch import nn


class CharDecoder(nn.Module):
    def __init__(self, num_embeddings: int,
                 hidden_size: int, padding_idx: int,
                 embedding_dim: int = 50) -> None:
        """
        Initialize the character level decoder
        Check notebooks: char-decoding.ipynb for explanation
        @param num_embeddings: number of embeddings.
            It equals vocab.tgt.length(tokens=False).
        @param hidden_size: hidden units for lstm layer.
        @param padding_idx: target char idx.
        @param embedding_dim: embedding dimension.
        """
        super(CharDecoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.char_decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=num_embeddings
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
        char_embedding = self.embedding(x)
        output, dec_state = self.char_decoder(
            char_embedding,
            dec_state
        )
        score = self.linear(output)
        return score, dec_state
