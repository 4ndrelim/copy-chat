#!/bin/sh
#SBATCH --job-name=pred_lora
#SBATCH --time=60
#SBATCH --mem-per-gpu=40G 
#SBATCH --gpus=a100-40:1
###SBATCH --nodelist=xgpg7

echo "SBATCH_INFO: Printing diagnostics (visible devices and nvidia-smi)..."
echo $CUDA_VISIBLE_DEVICES
srun nvidia-smi

echo "SBATCH_INFO: Running predict_slurm.py..."
python3 /home/c/czixuan/copy-chat/open-instruct/eval/predict_slurm.py \
--model /home/c/czixuan/copy-chat/models/merged/tweet/lora_tuned_Llama-3.1-8B \
--input_files /home/c/czixuan/copy-chat/open-instruct/datasets/formatted_datasets/prepared_sentiment_generation_short.jsonl  \
--output_file /home/c/czixuan/copy-chat/predictions/sentiment_generation/llama_old_sen_tuned_sen_gen.jsonl \
--use_vllm \
--use_chat_format

