# Databricks notebook source
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sys
sys.path.append('..')

from typing import List, Tuple

# COMMAND ----------

# MAGIC %md
# MAGIC From Luong and Manning (2016)
# MAGIC > The main idea is that when our word-level decoder produces an <UNK\> token, we run our character-level decoder (which you can think of as a character-level conditional language model) to instead generate the target word one character at a time.
# MAGIC 
# MAGIC <img src="images/char-nmt-luong-manning.png" />   
# MAGIC 
# MAGIC -- From assignment 5 --  
# MAGIC We have to deal with three parts of the character level decoder:  
# MAGIC 
# MAGIC > When we train the NMT system, we train the character decoder on every word in the target sentence (not just the words represented by <UNK\>).
# MAGIC 
# MAGIC 1. **Forward pass**: 
# MAGIC     * Get input character sequences - convert to character embeddings.
# MAGIC     * Run through unidirectional LSTM to get hidden states and cell states.
# MAGIC         * The initial hidden states are set to combined output from decoder. 
# MAGIC     * For each timestep, compute logits (using char_decoder W and char_decoder bias). Logits dimension `d = vocab.tgt.length(tokens=False)`.
# MAGIC 2. **Forward pass during training**:  
# MAGIC     * We do a forward pass with an input character sequence and get logits
# MAGIC     * Compare with target sequence and minimize using cross-entropy loss. Sum it up for a batch. 
# MAGIC     * Add loss of to word-based decoder loss   
# MAGIC 3. **During Test: Greedy decoding**:
# MAGIC     * During test time - first produce translation from NMT token based system.
# MAGIC     * *IF* the translation contains an _<unk>_ then for those positions we use the character decpder using the `combined_output` from decoder to initialize.
# MAGIC     * Use greedy decoding algorithm.  
# MAGIC 
# MAGIC <img src="images/greedy_decoding.png">
# MAGIC 
# MAGIC Note: To maintain the design pattern that I have been following, I will defer the implementation of steps 2 and 3 in main NMT forward. 

# COMMAND ----------

class CharDecoder(nn.Module):
    def __init__(self, num_embeddings: int,
                 hidden_size: int, padding_idx: int,
                 embedding_dim: int) -> None:
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


# COMMAND ----------


