DIRNAME=$(pwd)

echo ${DIRNAME}

DATASETS_DIR=${DIRNAME}/nmt/datasets
VOCAB_DIR=${DATASETS_DIR}/data
ENESDATA=${DATASETS_DIR}/data/en_es_data
WEIGHTS_DIR=${DIRNAME}/nmt/weights

mkdir -p ${WEIGHTS_DIR}

pipenv run python -m nmt --train --use-chardecoder --train-src=${ENESDATA}/train.es  --dev-src=${ENESDATA}/dev.es \
    --train-tgt=${ENESDATA}/train.en --dev-tgt=${ENESDATA}/dev.en --vocab=${VOCAB_DIR}/vocab.json \
    --save-path=${WEIGHTS_DIR}/model.bin --cuda
