# Databricks notebook source
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sys
sys.path.append('..')

from nmt.datasets import Vocab, batch_iter
from nmt.networks import CharEmbedding, Encoder

from typing import List, Tuple

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up everything till before we need to use attention

# COMMAND ----------

## Sample data
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

## Setting up vocab
vocab = Vocab.build(sentences_words_src, sentences_words_tgt)

# COMMAND ----------

## Generating a batch
data = list(zip(sentences_words_src, sentences_words_tgt))
data_generator = batch_iter(
    data=data,
    batch_size=4,
    shuffle=True
)
batch_src, batch_tgt = next(data_generator)
print(batch_src)

# COMMAND ----------

## Getting source lengths for encoder
source_length = [len(sent) for sent in batch_src]
print(source_length)

# COMMAND ----------

## Preparing input and output tensors
char_tensors_src = vocab.src.to_tensor(batch_src, tokens=False)
char_tensors_tgt = vocab.tgt.to_tensor(batch_tgt, tokens=False)
print(f"src char tensor size = {char_tensors_src.size()}; tgt char tensor size = {char_tensors_tgt.size()}")

# COMMAND ----------

## Defining Encoder
encoder = Encoder(
    num_embeddings=vocab.src.length(tokens=False),
    embedding_dim=300,
    char_padding_idx=vocab.src.pad_char_idx,
    hidden_size=1024
)

# COMMAND ----------

## Getting encoder output
char_enc_hidden, (char_hidden, char_cell), char_enc_proj = encoder(char_tensors_src, source_length)
char_enc_hidden.shape, char_hidden.shape, char_cell.shape, char_enc_proj.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Note here: We have encoder output equivalent of 6 timesteps

# COMMAND ----------

## Component of decoder - Target embedding layer
target_embedding = CharEmbedding(
    num_embeddings=vocab.tgt.length(tokens=False),
    char_embedding_dim=50,
    embedding_dim=300,
    char_padding_idx=vocab.tgt.pad_char_idx    
)

# COMMAND ----------

## Component of decoder - decoder LSTM Cell
sample_decoder_cell = nn.LSTMCell(
    input_size=300 + 1024,
    hidden_size=1024, bias=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC When we decode, we do it one time-step at a time. 

# COMMAND ----------

y_0 = torch.split(char_tensors_tgt, 1, dim=0)[0]

# COMMAND ----------

print("Target tensor shape:", y_0.shape)
embedded_y_0 = target_embedding(y_0)
print("Target embedded shape:", embedded_y_0.shape)

# COMMAND ----------

o_prev = torch.zeros(4, 1024, device="cpu")

# COMMAND ----------

ybar_t = torch.cat([embedded_y_0.squeeze(dim=0), o_prev], dim=1)
print(ybar_t.shape)

# COMMAND ----------

dec_hidden, dec_cell = sample_decoder_cell(ybar_t, (char_hidden, char_cell)) # initial dec_state

# COMMAND ----------

dec_hidden.shape, dec_cell.shape

# COMMAND ----------

# MAGIC %md
# MAGIC We have encoder output corresponding to _all_ timesteps and decoder output corresponding to _one_ timestep.

# COMMAND ----------

# MAGIC %md
# MAGIC # General Attention

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is Attention? 
# MAGIC 
# MAGIC There are a million explanations of attention and my favorite interpretation 
# MAGIC given that attention essentially talks of information retention and retrieval is that 
# MAGIC attention is a mechanism for mapping a query to a value by picking all relevant keys and
# MAGIC doing a weighted combination of their corresponding values.
# MAGIC   
# MAGIC The following image shows the computation of attention and context vectors using dot product attention.
# MAGIC 
# MAGIC <img src="images/attention.png" />
# MAGIC 
# MAGIC 
# MAGIC We assume that values are given. If keys are not, then we just use the values again. 
# MAGIC   
# MAGIC ### What is general attention? 
# MAGIC 
# MAGIC Note that our decoder hidden values (query) are of size hidden_size = 1024 but encoder outputs are 2*hidden_size = 2048. There are many cases in which this could occur. Therefore, we use a linear layer to project the encoder outputs to match the decoder hidden values (we got it from the encoder).

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we will say the encoder output *s = char_enc_hidden = the set*  
# MAGIC The decoder hidden *h_i = query*

# COMMAND ----------

dec_hidden_unsqueezed = dec_hidden.unsqueeze(dim=2)

# COMMAND ----------

dec_hidden_unsqueezed.shape

# COMMAND ----------

# https://pytorch.org/docs/stable/generated/torch.bmm.html
# No key therefore we use value as both key and value
score = char_enc_proj.bmm(dec_hidden_unsqueezed)

# COMMAND ----------

score.shape

# COMMAND ----------

score = score.squeeze(dim=2)

# COMMAND ----------

score.shape

# COMMAND ----------

attention_weights = F.softmax(score, dim=1)

# COMMAND ----------

attention_weights.shape

# COMMAND ----------

attention_weights = attention_weights.unsqueeze(dim=1)

# COMMAND ----------

attention_weights.shape

# COMMAND ----------

# MAGIC %md
# MAGIC https://pytorch.org/docs/stable/generated/torch.bmm.html
# MAGIC > If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.

# COMMAND ----------

# Attention scores multiplied by valuess
context_vector = attention_weights.bmm(char_enc_hidden)

# COMMAND ----------

context_vector.shape

# COMMAND ----------

context_vector = context_vector.squeeze(dim=1)

# COMMAND ----------

context_vector.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Putting it together

# COMMAND ----------

def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              enc_masks: torch.Tensor = None) -> torch.Tensor:

    query_unsqueezed = query.unsqueeze(dim=2)
    score = key.bmm(query_unsqueezed)
    score = score.squeeze(dim=2)

    if enc_masks is not None:
        score.data.masked_fill_(
            enc_masks.bool(),
            -float('inf')
        )

    attention_weights = F.softmax(score, dim=1)
    attention_weights = attention_weights.unsqueeze(dim=1)

    context_vector = attention_weights.bmm(value)
    context_vector = context_vector.squeeze(dim=1)

    return attention_weights, context_vector

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait what is that mask thing? 
# MAGIC 
# MAGIC It's the encoder mask.  
# MAGIC 
# MAGIC Because of batch thing, some sentences will be shorter than the longest sentence with remaining words having pad indices. We don't want attention to focus on those parts and therefore we set the parts corresponding to pad_indices to -inf because softmax(-inf) = 0. 

# COMMAND ----------

attention_weights, context_vector = attention(dec_hidden, char_enc_proj, char_enc_hidden)

# COMMAND ----------

attention_weights.shape, context_vector.shape

# COMMAND ----------


