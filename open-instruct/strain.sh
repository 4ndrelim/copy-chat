#!/bin/sh
#SBATCH --job-name=train_lora
#SBATCH --time=180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --mem-per-gpu=80G 
#SBATCH --gpus=a100-80:1
###SBATCH --nodelist=xgph1

echo "SBATCH_INFO: Printing diagnostics (visible devices and nvidia-smi)..."
echo $CUDA_VISIBLE_DEVICES
srun nvidia-smi
#srun nvcc --version
#srun accelerate config default
#srun accelerate env

echo "SBATCH_INFO: Running tweet_lora_llama.sh..."
#srun sh ~/copy-chat/open-instruct/scripts/custom_scripts/tweet_lora_llama.sh tsad_08_train_vanilla
#srun sh ~/copy-chat/open-instruct/scripts/custom_scripts/tweet_lora_llama.sh tsad_08_train_augment
#srun sh ~/copy-chat/open-instruct/scripts/custom_scripts/tweet_lora_llama.sh tsad_08_train_preprocessed
srun sh ~/copy-chat/open-instruct/scripts/custom_scripts/tweet_lora_llama.sh tsad_08_train_preprocessed_augment

#echo "SBATCH_INFO: Running tweet_lora_qwen.sh..."
#srun sh ~/copy-chat/open-instruct/scripts/custom_scripts/tweet_lora_qwen.sh tsad_04_train_vanilla

