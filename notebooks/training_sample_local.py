# Databricks notebook source
!pip install -r ../requirements.txt

# COMMAND ----------

import torch
from torch import nn
import numpy as np
from pathlib import Path
torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import sys
sys.path.append("..")

from nmt.models import NMT
from nmt.datasets import read_corpus, batch_iter, Vocab
import tqdm

#mlflow
import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# datasets_dir: {type: string, default: /dbfs/dmnr-public}
# vocab_dir: {type: string, default: /dbfs/dmnr-public/nmt-char}
# enesdata: {type: string, default: /dbfs/dmnr-public/nmt-char/en_es_data}
# output_dir: {type: string, default: /dbfs/dmnr-public/nmt-char/outputs}
# weights_dir: {type: string, default: /dbfs/dmnr-public/nmt-char/weights}

data_loc = "/dbfs/dmnr-public/nmt-char"
en_es_data_loc = "/dbfs/dmnr-public/nmt-char/en_es_data"
train_data_src_path = "/dbfs/dmnr-public/nmt-char/en_es_data/train_tiny.es"
train_data_tgt_path = "/dbfs/dmnr-public/nmt-char/en_es_data/train_tiny.en"
dev_data_src_path = "/dbfs/dmnr-public/nmt-char/en_es_data/dev_tiny.es"
dev_data_tgt_path = "/dbfs/dmnr-public/nmt-char/en_es_data/dev_tiny.en"
vocab_path = "/dbfs/dmnr-public/nmt-char/vocab_tiny_q2.json"

# COMMAND ----------

train_src = read_corpus(train_data_src_path)
train_tgt = read_corpus(train_data_tgt_path, is_target=True)
vocab = Vocab.load(vocab_path)

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
model.train()

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

epoch = 0

# COMMAND ----------

!current_user()

# COMMAND ----------

client = MlflowClient()
experiment = client.get_experiment_by_name("/Users/cjagadeesan@dataminr.com/nmt-notebook-example")
# client.set_experiment_tag(experiment_id, "nlp.framework", "pytorch")
experiment_id = experiment.experiment_id

# COMMAND ----------

experiment = client.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

Path("/tmp/local_model/").mkdir(parents=True)

# COMMAND ----------

with mlflow.start_run(experiment_id = experiment_id) as run:
    for epoch in range(MAX_EPOCH):
        cum_loss = 0
        for i, (src_sents, tgt_sents) in enumerate(batch_iter((train_src, train_tgt), batch_size=BATCH_SIZE, shuffle=True)):
            optimizer.zero_grad()
            batch_size = len(src_sents)

            batch_loss = -model(src_sents, tgt_sents).sum()
            batch_loss /= batch_size
            cum_loss += batch_loss
            batch_loss.backward()

             # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        cum_loss /= len(train_src)
        print(f"Epoch: {str(epoch).zfill(3)} - Cumulative loss: {cum_loss}")
        mlflow.log_metric("cumulative_loss", float(cum_loss))
    model.save("/tmp/local_model/notebook_model.bin")
    mlflow.log_artifact("/tmp/local_model")

# COMMAND ----------


