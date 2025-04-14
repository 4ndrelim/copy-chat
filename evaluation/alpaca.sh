#!/bin/bash

# python3 evaluation/prepare_alpaca_data.py --input_csv predictions/mimicry_generation/trump_v4_qwen.csv

alpaca_eval --model_outputs evaluation/alpaca_eval_prepared_data/model_outputs.json \
            --reference_outputs evaluation/alpaca_eval_prepared_data/reference_outputs.json \
            --annotators_config /Users/marcus/Desktop/code/misc/copy-chat/evaluation/configs.yaml \
            --output_path results/alpaca_mimicry_eval
