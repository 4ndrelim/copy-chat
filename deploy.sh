#!/bin/bash

# Config
MODEL_NAME="Qwen-Qwen2.5-7B-Instruct" 
TOKENIZER="Qwen-Qwen2.5-7B-Instruct"  # required for base models
SERVED_NAME="my_model"
PRECISION_TYPE="bfloat16"  # bfloat16 default 
MAX_LEN=32768  # cap at context window of the model e.g. 65536 etc
CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PARALLELISM=4
PORT=8181
#########

model_path="/home/models/$MODEL_NAME"

echo -e "[$(date)] Deploying [$MODEL_NAME] locally with vllm.."

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python -m vllm.entrypoints.openai.api_server \
    --port "$PORT" \
    --model "$model_path" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_LEN" \
    --dtype "$PRECISION_TYPE" \
    --tensor-parallel-size "$MODEL_PARALLELISM"
