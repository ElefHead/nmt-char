DIRNAME=$(pwd)

echo ${DIRNAME}

DATASETS_DIR=${DIRNAME}/nmt/datasets
VOCAB_DIR=${DATASETS_DIR}/data
ENESDATA=${DATASETS_DIR}/data/en_es_data
OUTPUT_DIR=${DATASETS_DIR}/outputs
WEIGHTS_DIR=${DIRNAME}/nmt/weights

mkdir -p ${OUTPUT_DIR}

pipenv run python -m nmt --cuda --use-chardecoder --test-src=${ENESDATA}/test.es \
    --test-tgt=${ENESDATA}/test.en \
    --model-path=${WEIGHTS_DIR}/model.bin \
    --output-path=${OUTPUT_DIR}/test_outputs.txt
