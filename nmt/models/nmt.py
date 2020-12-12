import torch
from torch import nn

from nmt.datasets import Vocab
from nmt.networks import Encoder, Decoder, CharDecoder

from typing import List, Tuple


class NMT(nn.Module):
    def __init__(self, vocab: Vocab,
                 embedding_dim: int,
                 hidden_size: int,
                 num_encoder_layers: int = 2) -> None:
        super(NMT, self).__init__()
        self.vocab = vocab
        self.encoder = Encoder(
            num_embeddings=vocab.src.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.src.pad_char_idx,
            hidden_size=hidden_size,
        )
        self.decoder = Decoder(
            num_embeddings=vocab.tgt.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.tgt.pad_char_idx,
            hidden_size=hidden_size
        )
        self.target_layer = nn.Linear(
            in_features=hidden_size,
            out_features=len(vocab.tgt),
            bias=False
        )
        self.char_decoder = CharDecoder(
            num_embeddings=vocab.tgt.length(tokens=False),
            hidden_size=hidden_size,
            padding_idx=vocab.tgt.pad_char_idx
        )
        self.hidden_size = hidden_size
        self.current_device = None

    def forward(self,
                x: List[List[int]],
                y: List[List[int]],
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        src_length = [len(sent) for sent in x]

        src_tensor = self.vocab.src.to_tensor(
            x, tokens=False, device=self.device)
        tgt_tensor = self.vocab.tgt.to_tensor(
            y, tokens=False, device=self.device)

        tgt_tensor_noend = tgt_tensor[:-1]
        src_encoding, dec_state = self.encoder(src_tensor, src_length)

        batch_size, _, _ = src_encoding.size()

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        combined_outputs = []
        for y_t in torch.split(tgt_tensor_noend, 1, dim=0):
            o_prev, dec_state, _ = self.decoder(
                y_t, src_encoding, dec_state, o_prev)
            combined_outputs.append(o_prev)

        combined_outputs = torch.stack(combined_outputs, dim=0)

        logits = self.target_layer(combined_outputs)

        max_word_len = tgt_tensor.shape[-1]
        target_chars = tgt_tensor[1:].contiguous().view(-1, max_word_len)
        target_outputs = combined_outputs.view(-1, self.hidden_size)

        target_chars_oov = target_chars.t()
        rnn_states_oov = target_outputs.unsqueeze(0)

        char_logits, char_dec_state = self.char_decoder(
            target_chars_oov[:-1],
            (rnn_states_oov, rnn_states_oov)
        )

        return logits, char_logits

    @property
    def device(self) -> torch.device:
        if not self.current_device:
            self.current_device = next(self.parameters()).device
        return self.current_device
