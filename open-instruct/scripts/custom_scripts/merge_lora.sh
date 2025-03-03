MODEL_SIZE=1.57
MODEL_TOKENIZER_TEMPLATE_PATH=/home/pints/brewery/models/1.5-Pints-2K-v0.1
LORA_LAYERS=...
MERGED_MODEL=...

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
