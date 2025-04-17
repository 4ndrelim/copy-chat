#!/bin/sh
#SBATCH --job-name=pred_lora
#SBATCH --time=90
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --mem-per-gpu=40G 
#SBATCH --gpus=a100-40:1
###SBATCH --nodelist=xgpg7

echo "SBATCH_INFO: Printing diagnostics (visible devices and nvidia-smi)..."
echo $CUDA_VISIBLE_DEVICES
srun nvidia-smi

echo "SBATCH_INFO: Running predict_slurm.py..."
python3 $HOME/copy-chat/open-instruct/eval/predict_slurm.py \
--model $HOME/copy-chat/models/merged/tweet/p2_Llama-3.1-8B_v11 \
--input_files $HOME/copy-chat/open-instruct/datasets/formatted_datasets/tsad_08_test_preprocessed.jsonl \
--output_file $HOME/copy-chat/predictions/sentiment_generation/tsad_llama_v11_preprocessed_augment.jsonl \
--use_vllm \
--use_chat_format \
--max_new_tokens 256

#--model $HOME/copy-chat/models/meta-llama-Llama-3.1-8B-Instruct \
#--stop_token '<|system|>' \
