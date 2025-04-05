#!/bin/sh

# Check for at least 1 argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 train_file (in copy-chat/open-instruct/datasets/formatted_datasets)"
    exit 1
fi

# below flags will degrade gpu communication but is required if more than 1 training tasks are being done
# export CUDA_VISIBLE_DEVICES=0 # should be set by node
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
export PYTHONPATH=$(pwd)/open-instruct:$PYTHONPATH

MODEL_SIZE=8b
MODEL_NAME=Llama-3.1-8B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU)) # 8

# determine the output model name
base="p2_$MODEL_NAME"
suffix="_v"
counter=1
dirname="$base"

while [ -d "$HOME/copy-chat/models/merged/tweet/$dirname" ]; do
    dirname="${base}${suffix}$(printf "%02d" $counter)"
    counter=$((counter + 1))
done

MODEL_TOKENIZER_TEMPLATE_PATH=$HOME/copy-chat/models/meta-llama-Llama-3.1-8B-Instruct
TRAIN_FILE=$HOME/copy-chat/open-instruct/datasets/formatted_datasets/$1.jsonl
OUTPUT_PATH=$HOME/copy-chat/models/adaptors/$dirname
SBATCH_INFO_FILE=$OUTPUT_PATH/training_info.txt

mkdir $OUTPUT_PATH
touch $SBATCH_INFO_FILE
echo "Training $MODEL_NAME of size ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps" | tee -a $SBATCH_INFO_FILE
echo "SBATCH_INFO: Train file: $TRAIN_FILE" | tee -a $SBATCH_INFO_FILE
echo "SBATCH_INFO: Output path: $OUTPUT_PATH"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
    --use_flash_attn \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --tokenizer_name $MODEL_TOKENIZER_TEMPLATE_PATH \
    --chat_template_name $MODEL_TOKENIZER_TEMPLATE_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_PATH \

    # --with_tracking \
    # --logging_steps 1 \
    # --report_to wandb \
    # --enable_wandb True \
    # --wandb_project <name> \
    # --wandb_name <name> \
    # --wandb_entity <entity name>

# sh "$(dirname "$0")/merge_lora_llama.sh"
# Merge script copy pasted here:
LORA_LAYERS=$OUTPUT_PATH
MERGED_MODEL=$HOME/copy-chat/models/merged/tweet/$dirname
echo "SBATCH_INFO: Merged model path: $MERGED_MODEL"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    open_instruct/merge_lora.py \
    --base_model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
    --lora_model_name_or_path $LORA_LAYERS \
    --output_dir $MERGED_MODEL \
    --save_tokenizer

