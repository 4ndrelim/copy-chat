#!/bin/bash

# Stop script on any error
# set -e  

export HF_TOKEN="<YOUR_TOKEN>"

# Define a list of model names
MODELS=(
    "deepseek-ai/deepseek-math-7b-instruct"
)

# Define save directory
SAVE_DIR="/home/andre/andre/models"

# Define log file
LOG_FILE="download_log.txt"

# Loop through models and call the Python script
for model in "${MODELS[@]}"; do
    echo "--------------------------------" | tee -a "$LOG_FILE"
    echo "Downloading model: $model" | tee -a "$LOG_FILE"
    
    if python download.py --model_name "$model" --save_dir "$SAVE_DIR"; then
        echo "Successfully downloaded: $model" | tee -a "$LOG_FILE"
    else
        echo "Error downloading: $model. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    fi

    echo "--------------------------------" | tee -a "$LOG_FILE"
done
