# Databricks notebook source
import torch
from torch import nn
from pathlib import Path
import numpy as np
from collections import namedtuple

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import sys
sys.path.append("..")

from nmt.models import NMT
from nmt.datasets import read_corpus, batch_iter, Vocab
import tqdm

# COMMAND ----------

data_loc = Path("..") / "nmt" / "datasets" / "data"
en_es_data_loc = data_loc / "en_es_data"
train_data_src_path = en_es_data_loc / "train_tiny.es"
train_data_tgt_path = en_es_data_loc / "train_tiny.en"
dev_data_src_path = en_es_data_loc / "dev_tiny.es"
dev_data_tgt_path = en_es_data_loc / "dev_tiny.en"
vocab_path = data_loc / "vocab_tiny_q2.json"

# COMMAND ----------

train_src = read_corpus(train_data_src_path)
train_tgt = read_corpus(train_data_tgt_path, is_target=True)
vocab = Vocab.load(vocab_path)

# COMMAND ----------

len(vocab.src), len(vocab.tgt)

# COMMAND ----------

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

# COMMAND ----------

valid_src = read_corpus(dev_data_src_path)
valid_tgt = read_corpus(dev_data_tgt_path, is_target=True)

# COMMAND ----------

BATCH_SIZE=2
MAX_EPOCH=201
SEED=42
EMBEDDING_SIZE=256
HIDDEN_SIZE=256
GRAD_CLIP=5.0
UNIFORM_INIT=0.1
USE_CHAR_DECODER=True
LEARNING_RATE=0.001

# COMMAND ----------

model = NMT(
    vocab=vocab,
    embedding_dim=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    use_char_decoder=True
)

# COMMAND ----------

model = model.train()

# COMMAND ----------

uniform_init = UNIFORM_INIT
if np.abs(uniform_init) > 0.:
    print('uniformly initialize parameters [-%f, +%f]' %
            (uniform_init, uniform_init), file=sys.stderr)
    for p in model.parameters():
        p.data.uniform_(-uniform_init, uniform_init)

# COMMAND ----------

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# COMMAND ----------

example = [train_src[0]] ## Source sent

# COMMAND ----------

beam_size = 5

# COMMAND ----------

example_char_tensor = model.vocab.src.to_tensor(example, tokens=False)

# COMMAND ----------

src_encode, dec_state = model.encoder(example_char_tensor, [len(x) for x in example])

# COMMAND ----------

hypotheses = [['<s>']]

# COMMAND ----------

completed_hypotheses = []

# COMMAND ----------

hyp_scores = torch.zeros(
    len(hypotheses), dtype=torch.float
)
print(hyp_scores.size())

# COMMAND ----------

num_hyps = len(hypotheses)

# COMMAND ----------

init_attention = torch.zeros(1, model.hidden_size)

# COMMAND ----------

end_of_sent = model.vocab.tgt.end_token_idx

# COMMAND ----------

#### Iterating through 1 timestep

# COMMAND ----------

o_prev = torch.zeros(1, model.hidden_size, device=model.device)

# COMMAND ----------

y_t = model.vocab.tgt.to_tensor(
             list([hyp[-1]] for hyp in hypotheses),
             tokens=False, device=model.device
        )


# COMMAND ----------

y_t.shape

# COMMAND ----------

o_prev, dec_state, _ = model.decoder(y_t, src_encode, dec_state, o_prev, None)

# COMMAND ----------

log_p_t = model.generator(o_prev)

# COMMAND ----------

log_p_t.shape

# COMMAND ----------

log_p_t

# COMMAND ----------

live_hyp_num = beam_size - len(completed_hypotheses)

# COMMAND ----------

continuing_hyp_scores = (hyp_scores.unsqueeze(
                1).expand_as(log_p_t) + log_p_t).view(-1)
continuing_hyp_scores.shape

# COMMAND ----------

top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                continuing_hyp_scores, k=live_hyp_num)

# COMMAND ----------

top_cand_hyp_scores, top_cand_hyp_pos

# COMMAND ----------

prev_hyp_ids = top_cand_hyp_pos / len(model.vocab.tgt)
hyp_word_ids = top_cand_hyp_pos % len(model.vocab.tgt)
print(prev_hyp_ids, hyp_word_ids)

# COMMAND ----------

len(model.vocab.tgt)

# COMMAND ----------

top_cand_hyp_pos / 32

# COMMAND ----------

new_hypotheses = []
live_hyp_ids = []
new_hyp_scores = []

# COMMAND ----------

decoderStatesForUNKsHere = []

# COMMAND ----------

for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
    prev_hyp_id = prev_hyp_id.item()
    hyp_word_id = hyp_word_id.item()
    cand_new_hyp_score = cand_new_hyp_score.item()

    hyp_word = model.vocab.tgt.to_tokens(hyp_word_id)

    # Record output layer in case UNK was generated
    if hyp_word == "<unk>":
        hyp_word = "<unk>"+str(len(decoderStatesForUNKsHere))
        decoderStatesForUNKsHere.append(o_prev[prev_hyp_id])

    new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
    if hyp_word == '</s>':
        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                score=cand_new_hyp_score))
    else:
        new_hypotheses.append(new_hyp_sent)
        live_hyp_ids.append(prev_hyp_id)
        new_hyp_scores.append(cand_new_hyp_score)

# COMMAND ----------

print(new_hypotheses)
print(live_hyp_ids)
print(new_hyp_scores)
print(completed_hypotheses)

# COMMAND ----------

if len(decoderStatesForUNKsHere) > 0 and model.char_decoder is not None:  # decode UNKs
    decoderStatesForUNKsHere = torch.stack(
        decoderStatesForUNKsHere, dim=0)
    decodedWords = model.greedy_char_decode((decoderStatesForUNKsHere.unsqueeze(
        0), decoderStatesForUNKsHere.unsqueeze(0)), max_length=21)
    assert len(decodedWords) == decoderStatesForUNKsHere.size()[
        0], "Incorrect number of decoded words"
    for hyp in new_hypotheses:
        if hyp[-1].startswith("<unk>"):
            hyp[-1] = decodedWords[int(hyp[-1][5:])]  # [:-1]


# COMMAND ----------

live_hyp_ids = torch.tensor(
    live_hyp_ids,
    dtype=torch.long,
    device=model.device
)

# COMMAND ----------

o_prev[live_hyp_ids].shape

# COMMAND ----------

hypotheses = new_hypotheses
hyp_scores = torch.tensor(
    new_hyp_scores, dtype=torch.float, device=model.device)


# COMMAND ----------

if len(completed_hypotheses) == 0:
    completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                            score=hyp_scores[0].item()))

# COMMAND ----------

completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

# COMMAND ----------

completed_hypotheses

# COMMAND ----------

hypotheses = new_hypotheses

# COMMAND ----------

hypotheses

# COMMAND ----------

# MAGIC %%time
# MAGIC for epoch in range(MAX_EPOCH):
# MAGIC     cum_loss = 0
# MAGIC     for i, (src_sents, tgt_sents) in enumerate(batch_iter((train_src, train_tgt), batch_size=BATCH_SIZE, shuffle=True)):
# MAGIC         optimizer.zero_grad()
# MAGIC         batch_size = len(src_sents)
# MAGIC 
# MAGIC         batch_loss = -model(src_sents, tgt_sents).sum()
# MAGIC         batch_loss /= batch_size
# MAGIC         cum_loss += batch_loss
# MAGIC         batch_loss.backward()
# MAGIC 
# MAGIC          # clip gradient
# MAGIC         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
# MAGIC         optimizer.step()
# MAGIC     cum_loss /= len(train_src)
# MAGIC     print(f"Epoch: {str(epoch).zfill(3)} - Cumulative loss: {cum_loss}")

# COMMAND ----------

from nmt.scripts import beam_search_decoder

# COMMAND ----------

hypothesis = beam_search_decoder(
    model=model,
    src_sent=train_src[0]
)

# COMMAND ----------

hypothesis

# COMMAND ----------


