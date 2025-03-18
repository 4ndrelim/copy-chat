#!/bin/sh
#SBATCH --job-name=train_lora
#SBATCH --gpus=1
#SBATCH --nodelist=xgpf3

#ls scripts/custom_scripts
#nvidia-smi
srun scripts/custom_scripts/books_lora.sh
