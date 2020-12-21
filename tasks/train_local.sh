DIRNAME=$(pwd)

echo ${DIRNAME}

DATASETS_DIR=${DIRNAME}/nmt/datasets
VOCAB_DIR=${DATASETS_DIR}/data
ENESDATA=${DATASETS_DIR}/data/en_es_data
WEIGHTS_DIR=${DIRNAME}/nmt/weights

mkdir -p ${WEIGHTS_DIR}

pipenv run python -m nmt --train --use-chardecoder --train-src=${ENESDATA}/train_tiny.es  --dev-src=${ENESDATA}/dev_tiny.es \
    --train-tgt=${ENESDATA}/train_tiny.en --dev-tgt=${ENESDATA}/dev_tiny.en --vocab=${VOCAB_DIR}/vocab_tiny_q2.json \
    --batch-size=2 --max-epoch=201 --valid-niter=100 --save-path=${WEIGHTS_DIR}/local_model.bin
