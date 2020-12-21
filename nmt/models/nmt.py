import torch
from torch import nn

from nmt.datasets import Vocab
from nmt.networks import Encoder, Decoder, CharDecoder
from nmt.layers import Generator
from typing import List, Tuple


class NMT(nn.Module):
    def __init__(self, vocab: Vocab,
                 embedding_dim: int,
                 hidden_size: int,
                 dropout_prob: float = 0.3,
                 use_char_decoder: bool = False) -> None:
        super(NMT, self).__init__()
        self.use_char_decoder = use_char_decoder
        self.vocab = vocab
        self.dropout_prob = dropout_prob
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            num_embeddings=vocab.src.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.src.pad_char_idx,
            hidden_size=hidden_size
        )
        self.decoder = Decoder(
            num_embeddings=vocab.tgt.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.tgt.pad_char_idx,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob
        )
        self.generator = Generator(
            in_features=hidden_size,
            out_features=len(vocab.tgt)
        )
        self.char_decoder = None
        if self.use_char_decoder:
            self.char_decoder = CharDecoder(
                num_embeddings=vocab.tgt.length(tokens=False),
                hidden_size=hidden_size,
                padding_idx=vocab.tgt.pad_char_idx
            )
        self.hidden_size = hidden_size
        self.current_device = None

    def forward(self,
                x: List[List[int]],
                y: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:

        src_length = [len(sent) for sent in x]

        src_tensor = self.vocab.src.to_tensor(
            x, tokens=False, device=self.device
        )
        tgt_tensor = self.vocab.src.to_tensor(
            y, tokens=False, device=self.device
        )
        tgt_token_tensor = self.vocab.tgt.to_tensor(
            y, tokens=True, device=self.device
        )

        tgt_tensor_noend = tgt_tensor[:-1]
        src_encoding, dec_state, src_enc_projection = self.encoder(
            src_tensor, src_length
        )

        enc_masks = self.generate_sentence_masks(src_encoding, src_length)

        batch_size, _, _ = src_encoding.size()

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        combined_outputs = []
        for y_t in torch.split(tgt_tensor_noend, 1, dim=0):
            o_prev, dec_state, _ = self.decoder(
                y_t, src_encoding, dec_state, src_enc_projection,
                o_prev, enc_masks)
            combined_outputs.append(o_prev)

        combined_outputs = torch.stack(combined_outputs, dim=0)

        probs = self.generator(combined_outputs)

        # zero out the pad targets
        target_masks = (tgt_token_tensor != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_token_log_prob = torch.gather(
            probs, index=tgt_token_tensor[1:].unsqueeze(-1), dim=-1
        ).squeeze(-1) * target_masks[1:]

        loss = target_token_log_prob.sum()

        if self.use_char_decoder:
            max_word_len = tgt_tensor.shape[-1]
            target_chars = tgt_tensor[1:].contiguous().view(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, self.hidden_size)

            target_chars_oov = target_chars.t()
            rnn_states_oov = target_outputs.unsqueeze(0)

            char_logits, char_dec_state = self.char_decoder(
                target_chars_oov[:-1],
                (rnn_states_oov, rnn_states_oov)
            )
            char_logits = char_logits.permute(1, 2, 0)
            # char_logits = char_logits.view(-1, char_logits.shape[-1])
            target_chars_oov = target_chars_oov[1:].permute(1, 0)

            char_loss = nn.CrossEntropyLoss(
                reduction="sum"
            )

            loss = loss - char_loss(char_logits, target_chars_oov)

        return loss

    def generate_sentence_masks(self,
                                enc_out: torch.Tensor,
                                source_lengths: List[int]) -> torch.Tensor:
        enc_masks = torch.zeros(
            enc_out.size(0),
            enc_out.size(1),
            device=self.device
        )
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks

    def greedy_char_decode(self, dec_state, max_length: int = 21):
        batch_size = dec_state[0].size(1)

        start = self.vocab.tgt.start_char_idx

        decoded_lists = [""] * batch_size
        inp = torch.tensor(
            [[start] * batch_size],
            device=self.device
        )
        current_states = dec_state

        for _ in range(max_length):
            scores, current_states = self.char_decoder(
                inp,
                current_states
            )
            probabilities = torch.softmax(
                scores.squeeze(0), dim=1
            )
            inp = torch.argmax(
                probabilities, dim=1
            ).unsqueeze(0)

            for i, c in enumerate(inp.squeeze(dim=0)):
                decoded_lists[i] += self.vocab.tgt.to_char(int(c))

        decoded_words = []
        for word in decoded_lists:
            end_pos = word.find(self.vocab.tgt.end_char)
            decoded_words.append(word if end_pos == -1 else word[: end_pos])

        return decoded_words

    @property
    def device(self) -> torch.device:
        if not self.current_device:
            self.current_device = next(self.parameters()).device
        return self.current_device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print(f'save model parameters to [{path}]')

        params = {
            'args': dict(
                hidden_size=self.hidden_size,
                dropout_prob=self.dropout_prob,
                use_char_decoder=self.use_char_decoder,
                embedding_dim=self.embedding_dim
            ),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
