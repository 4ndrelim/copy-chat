#!/bin/bash

# python3 evaluation/prepare_alpaca_data.py --input_csv predictions/mimicry_generation/trump_v4_llama.csv --output_dir evaluation/alpaca_eval_prepared_data/trump_v4_llama --generator_name trump_v4_llama

CURRENT_DIR=$(pwd)

alpaca_eval --model_outputs evaluation/alpaca_eval_prepared_data/trump_v4_llama/model_outputs.json \
            --reference_outputs evaluation/alpaca_eval_prepared_data/trump_v4_llama/reference_outputs.json \
            --annotators_config $CURRENT_DIR/evaluation/configs.yaml \
            --output_path results/alpaca_mimicry_eval/trump_v4_llama
