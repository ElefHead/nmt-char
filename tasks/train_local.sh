DIRNAME=$(pwd)

DATASETS_DIR=${DIRNAME}/datasets
VOCAB_DIR=${DATASETS_DIR}/vocab
ENESDATA=${DATASETS_DIR}/data/en_es_data
OUTPUT_DIR=${DIRNAME}/datasets/outputs

mkdir -p OUTPUT_DIR
touch ${OUTPUT_DIR}/testoutputs_local_run.txt
pipenv run python -m nmt --train --train-src=${ENESDATA}/train_tiny.es  --dev-src=${ENESDATA}/dev_tiny.es \
    --train-tgt=${ENESDATA}/train_tiny.en --dev-tgt=${ENESDATA}/dev_tiny.en --vocab=${VOCAB_DIR}/vocab_tiny_q2.json \
    --batch-size=2 --max-epoch=201 --valid-niter=100
