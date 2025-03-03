# below flags will degrade gpu communication but is required if more than 1 training tasks are being done
export CUDA_VISIBLE_DEVICES=6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONPATH=$(pwd)/open-instruct:$PYTHONPATH

MODEL_SIZE=8b
MODEL_NAME=LLama3.1-8b
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU)) # 8
MODEL_TOKENIZER_TEMPLATE_PATH=/home/andre/andre/models/meta-llama-Llama-3.1-8B-Instruct
TRAIN_FILE=/home/andre/andre/ml-frameworks/open-instruct/datasets/formatted_datasets/prepared_books_data.jsonl
OUTPUT_PATH=/home/andre/andre/models/adaptors/books_$MODEL_NAME/
echo "Training $MODEL_NAME of size ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
    --use_flash_attn \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name $MODEL_TOKENIZER_TEMPLATE_PATH \
    --chat_template_name $MODEL_TOKENIZER_TEMPLATE_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 4e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --output_dir $OUTPUT_PATH \

    # --with_tracking \
    # --logging_steps 1 \
    # --report_to wandb \
    # --enable_wandb True \
    # --wandb_project <name> \
    # --wandb_name <name> \
    # --wandb_entity <entity name>
    

# for merging
# python open_instruct/merge_lora.py \
#     --base_model_name_or_path $MODEL_TOKENIZER_TEMPLATE_PATH \
#     --lora_model_name_or_path $OUTPUT_PATH \
#     --output_dir /home/andre/andre/models/${MODEL_NAME}_lora_merged/
#     --save_tokenizer
