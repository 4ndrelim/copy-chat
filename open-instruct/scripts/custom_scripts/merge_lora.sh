MODEL_SIZE=8b
MODEL_TOKENIZER_TEMPLATE_PATH=/home/andre/andre/models/meta-llama-Llama-3.1-8B-Instruct
LORA_LAYERS=/home/andre/andre/models/adaptors/tweet_LLama3.1-8b
MERGED_MODEL=/home/andre/andre/models/merged/tweet/lora_tuned_llama3.1_8b

export CUDA_VISIBLE_DEVICES=6

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    open_instruct/merge_lora.py \
    --base_model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
    --lora_model_name_or_path $LORA_LAYERS \
    --output_dir $MERGED_MODEL \
    --save_tokenizer
