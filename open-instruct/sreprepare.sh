#!/bin/sh

set -e

# Check for at least 1 argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 dataset_name"
    exit 1
fi

SYS_PROMPT=$(python -c 'import json; print(json.load(open("dataset_preparation/prompt_templates/tweet_sentiment_generation.json"))["system_prompt"])')
echo "Using system_prompt:\n$SYS_PROMPT"

python dataset_preparation/tsad_sentiment_generation_preparer.py --dataset_name $1 --input_path ../dataset/tweet_sentiment/train/zixuan_andre/train_vanilla.jsonl --template_path dataset_preparation/prompt_templates/tweet_sentiment_generation.json --output_path datasets/formatted_datasets/$1_train_vanilla.jsonl
echo "Wrote train_vanilla dataset to datasets/formatted_datasets/$1_train_vanilla.jsonl"
python dataset_preparation/tsad_sentiment_generation_preparer.py --dataset_name $1 --input_path ../dataset/tweet_sentiment/train/zixuan_andre/train_vanilla_augment.jsonl --template_path dataset_preparation/prompt_templates/tweet_sentiment_generation.json --output_path datasets/formatted_datasets/$1_train_augment.jsonl
echo "Wrote train_vanilla_augment dataset to datasets/formatted_datasets/$1_train_augment.jsonl"
python dataset_preparation/tsad_sentiment_generation_preparer.py --dataset_name $1 --input_path ../dataset/tweet_sentiment/test/test.jsonl --template_path dataset_preparation/prompt_templates/tweet_sentiment_generation.json --output_path datasets/formatted_datasets/$1_test.jsonl --is_evaluation true
echo "Wrote test dataset to datasets/formatted_datasets/$1_test.jsonl"

