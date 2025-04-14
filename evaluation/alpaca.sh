#!/bin/bash

MODEL_NAME="trump_v4_llama"
CURRENT_DIR=$(pwd)

# python3 evaluation/prepare_alpaca_data.py --input_csv predictions/mimicry_generation/${MODEL_NAME}.csv --output_dir evaluation/alpaca_eval_prepared_data/${MODEL_NAME} --generator_name ${MODEL_NAME}

alpaca_eval --model_outputs evaluation/alpaca_eval_prepared_data/${MODEL_NAME}/model_outputs.json \
            --reference_outputs evaluation/alpaca_eval_prepared_data/${MODEL_NAME}/reference_outputs.json \
            --annotators_config $CURRENT_DIR/evaluation/configs.yaml \
            --output_path evaluation/EvalResults/AlpacaMimicry/${MODEL_NAME}
