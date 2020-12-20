DIRNAME=$(pwd)

echo ${DIRNAME}

DATASETS_DIR=${DIRNAME}/nmt/datasets
VOCAB_DIR=${DATASETS_DIR}/data
ENESDATA=${DATASETS_DIR}/data/en_es_data
OUTPUT_DIR=${DATASETS_DIR}/outputs
WEIGHTS_DIR=${DIRNAME}/nmt/weights

mkdir -p ${OUTPUT_DIR}
mkdir -p ${WEIGHTS_DIR}

touch ${OUTPUT_DIR}/testoutputs_local_run.txt
pipenv run python -m nmt --use-chardecoder --test-src=${ENESDATA}/test_tiny.es \
    --test-tgt=${ENESDATA}/test_tiny.en \
    --model-path=${WEIGHTS_DIR}/local_model.bin \
    --output-path=${OUTPUT_DIR}/test_outputs_local.txt
