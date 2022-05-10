# Databricks notebook source
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple

import sys
sys.path.append("..")

from nmt.networks import Encoder, Decoder, CharDecoder
from nmt.datasets import Vocab
from nmt.layers import Generator
from nmt.datasets import batch_iter

# COMMAND ----------

# MAGIC %md
# MAGIC The goal in this notebook is to sort of recreate assignment 4. We have almost everything required. 

# COMMAND ----------

class NMT(nn.Module):
    def __init__(self, vocab: Vocab,
                 embedding_dim: int,
                 hidden_size: int,
                 num_encoder_layers: int = 2,
                 use_char_decoder: bool = False) -> None:
        super(NMT, self).__init__()
        self.vocab = vocab
        self.encoder = Encoder(
            num_embeddings=vocab.src.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.src.pad_char_idx,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers
        )
        self.decoder = Decoder(
            num_embeddings=vocab.tgt.length(tokens=False),
            embedding_dim=embedding_dim,
            char_padding_idx=vocab.tgt.pad_char_idx,
            hidden_size=hidden_size
        )
        self.generator = Generator(
            in_features=hidden_size,
            out_features=len(vocab.tgt)
        )
        self.char_decoder = None
        if use_char_decoder:
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
            x, tokens=False, device=self.device
        )
        tgt_tensor = self.vocab.tgt.to_tensor(
            y, tokens=False, device=self.device
        )
        tgt_token_tensor = self.vocab.tgt.to_tensor(
            y, tokens=True, device=self.device
        )

        tgt_tensor_noend = tgt_tensor[:-1]
        src_encoding, dec_state = self.encoder(src_tensor, src_length)

        enc_masks = self.generate_sentence_masks(src_encoding, src_length)

        batch_size, _, _ = src_encoding.size()

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        combined_outputs = []
        for y_t in torch.split(tgt_tensor_noend, 1, dim=0):
            o_prev, dec_state, _ = self.decoder(
                y_t, src_encoding, dec_state, o_prev, enc_masks)
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

        if self.char_decoder:
            max_word_len = tgt_tensor.shape[-1]
            target_chars = tgt_tensor[1:].contiguous().view(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, self.hidden_size)

            target_chars_oov = target_chars.t()
            rnn_states_oov = target_outputs.unsqueeze(0)

            char_logits, char_dec_state = self.char_decoder(
                target_chars_oov[:-1],
                (rnn_states_oov, rnn_states_oov)
            )

            char_logits = char_logits.view(-1, char_logits.shape[-1])
            target_chars_oov = target_chars_oov[1:].contiguous().view(-1)

            char_loss = nn.CrossEntropyLoss(
                reduction="sum",
                ignore_index=self.vocab.tgt.pad_char_idx
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

        start = self.vocab.tgt.char_start_idx
        end = self.vocab.tgt.char_end_idx

        output_words = [] * batch_size
        start_char_ids = [[start] * batch_size]
        current_char_ids = torch.tensor(
            start_char_ids, device=self.device
        )
        current_states = dec_state 

        for _ in range(max_length):
            score, current_states = self.char_decoder(current_char_ids, current_states)
            prob = F.softmax(score, dim=1)
            current_char_ids = prob.argmax(dim=1).unsqueeze(dim=0)
            for i, c in enumerate(current_char_ids.squeeze(dim=0)):
                output_words[i] += self.vocab.tgt.to_char(int(c))

        decoded_words = []
        for word in output_words:
            end_pos = word.find(self.vocab.tgt.to_char(end))
            decoded_words.append(word if end_pos == -1 else word[:end_pos])

        return decoded_words

    @property
    def device(self) -> torch.device:
        if not self.current_device:
            self.current_device = next(self.parameters()).device
        return self.current_device

# COMMAND ----------

sentences_words_src = [
    ['Human:', 'What', 'do', 'we', 'want?'],
    ['Computer:', 'Natural', 'language', 'processing!'],
    ['Human:', 'When', 'do', 'we', 'want', 'it?'],
    ['Computer:', 'When', 'do', 'we', 'want', 'what?']
]

sentences_words_tgt = [
    ['<s>', 'Human:', 'What', 'do', 'we', 'want?', '</s>'],
    ['<s>', 'Computer:', 'Natural', 'language', 'processing!', '</s>'],
    ['<s>', 'Human:', 'When', 'do', 'we', 'want', 'it?', '</s>'],
    ['<s>', 'Computer:', 'When', 'do', 'we', 'want', 'what?', '</s>']
]

# COMMAND ----------

vocab = Vocab.build(sentences_words_src, sentences_words_tgt)

# COMMAND ----------

data = list(zip(sentences_words_src, sentences_words_tgt))
data_generator = batch_iter(
    data=data,
    batch_size=4,
    shuffle=True
)
batch_src, batch_tgt = next(data_generator)

# COMMAND ----------

model = NMT(
    vocab=vocab, 
    embedding_dim=300,
    hidden_size=1024
).cuda()

# COMMAND ----------

loss = model(batch_src, batch_tgt)

# COMMAND ----------

print(loss)

# COMMAND ----------

len(list(model.parameters()))

# COMMAND ----------

model_char = NMT(
    vocab=vocab, 
    embedding_dim=300,
    hidden_size=1024,
    use_char_decoder=True
).cuda()

# COMMAND ----------

loss_char = model(batch_src, batch_tgt)

# COMMAND ----------

print(loss_char)

# COMMAND ----------

len(list(model_char.parameters()))

# COMMAND ----------


