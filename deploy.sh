#!/bin/bash

# # Config
# # MODEL_NAME="Llama-3.1-8B-Instruct"
# # TOKENIZER="Llama-3.1-8B-Instruct"
# MODEL_NAME="lora_tuned_llama3.1_8b"
# TOKENIZER="lora_tuned_llama3.1_8b"  # required for base models
# SERVED_NAME="my_model"
# PRECISION_TYPE="bfloat16"  # bfloat16 default fp32 
# MAX_LEN=8192  # cap at context window of the model
# CUDA_VISIBLE_DEVICES=2
# MODEL_PARALLELISM=1
# PORT=8383
# #########

# # model_path="/disk1/nuochen/models/$MODEL_NAME"
# model_path="/home/andre/andre/models/merged/tweet/$MODEL_NAME"

# echo -e "[$(date)] Deploying [$MODEL_NAME] locally with vllm.."

# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# python -m vllm.entrypoints.openai.api_server \
#     --port "$PORT" \
#     --model "$model_path" \
#     --served-model-name "$SERVED_NAME" \
#     --max-model-len "$MAX_LEN" \
#     --dtype "$PRECISION_TYPE" \
#     --tensor-parallel-size "$MODEL_PARALLELISM"



# Config
MODEL_NAME="trump_v1"
TOKENIZER="trump_v1"
# MODEL_NAME="Qwen-Qwen2.5-7B"
# TOKENIZER="Qwen-Qwen2.5-7B"  # required for base models
# MODEL_NAME="lora_tuned_Qwen2.5-7B-Instruct"
# TOKENIZER="lora_tuned_Qwen2.5-7B-Instruct"  # required for base models
# MODEL_NAME="lora_tuned_sentiment_generation_Qwen2.5-7B-Instruct"
# TOKENIZER="lora_tuned_sentiment_generation_Qwen2.5-7B-Instruct"  # required for base models
SERVED_NAME="my_qwen_model"
PRECISION_TYPE="bfloat16"  # bfloat16 default fp32 
MAX_LEN=2048  # cap at context window of the model
CUDA_VISIBLE_DEVICES=6
MODEL_PARALLELISM=1
PORT=8282
#########

# model_path="/shared/ssd/models/$MODEL_NAME"
# model_path="/home/andre/andre/models/merged/tweet/$MODEL_NAME"
model_path="/home/andre/andre/models/merged/$MODEL_NAME"
# model_path="/home/andre/andre/models/$MODEL_NAME"

echo -e "[$(date)] Deploying [$MODEL_NAME] locally with vllm.."

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python -m vllm.entrypoints.openai.api_server \
    --port "$PORT" \
    --model "$model_path" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_LEN" \
    --dtype "$PRECISION_TYPE" \
    --tensor-parallel-size "$MODEL_PARALLELISM"
    