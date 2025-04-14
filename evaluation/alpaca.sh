#!/bin/bash

python3 evaluation/prepare_alpaca_data.py --input_csv predictions/mimicry_generation/trump_v4_qwen.csv --output_dir evaluation/alpaca_eval_prepared_data/trump_v4_qwen_mimicry --generator_name trump_v4_qwen_mimicry

alpaca_eval --model_outputs evaluation/alpaca_eval_prepared_data/trump_v4_qwen_mimicry/model_outputs.json \
            --reference_outputs evaluation/alpaca_eval_prepared_data/trump_v4_qwen_mimicry/reference_outputs.json \
            --annotators_config /Users/marcus/Desktop/code/misc/copy-chat/evaluation/configs.yaml \
            --output_path results/alpaca_mimicry_eval/trump_v4_qwen_mimicry

            # --annotators_config weighted_alpaca_eval_gpt4_turbo \
