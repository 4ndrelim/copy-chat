MODEL_SIZE=8b
MODEL_NAME=Llama-3.1-8B
MODEL_TOKENIZER_TEMPLATE_PATH=$HOME/copy-chat/models/meta-llama-Llama-3.1-8B-Instruct
LORA_LAYERS=$HOME/copy-chat/models/adaptors/sentiment_generation_$MODEL_NAME
MERGED_MODEL=$HOME/copy-chat/models/merged/tweet/lora_tuned_sen_gen_$MODEL_NAME

#export CUDA_VISIBLE_DEVICES=6

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    open_instruct/merge_lora.py \
    --base_model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
    --lora_model_name_or_path $LORA_LAYERS \
    --output_dir $MERGED_MODEL \
    --save_tokenizer
