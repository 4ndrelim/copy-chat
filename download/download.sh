#!/bin/bash

# Stop script on any error
# set -e  

export HF_TOKEN="<YOUR_TOKEN>"

# Define a list of model names
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# Define save directory
SAVE_DIR="/home/models"

# Define log file
LOG_FILE="download_log.txt"

# Loop through models and call the Python script
for model in "${MODELS[@]}"; do
    echo "------------------------------------------------------------------------------------------------"
    echo "Downloading model: $model"
    
    if python download.py --model_name "$model" --save_dir "$SAVE_DIR"; then
        echo "Successfully downloaded: $model"
    else
        echo "Error downloading: $model. Check $LOG_FILE for details." 
    fi

    echo "------------------------------------------------------------------------------------------------"
done
