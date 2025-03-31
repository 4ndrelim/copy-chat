"""
This script converts existing dataset to a format that can be parsed by allenAI Instruct
"""

from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
from logging import Logger
import uuid
import json
from jsonargparse import CLI

from dataset_preparation.dataset_utils import (
    load_jsonl_file,
    write_jsonl_file,
)

from utils.logger import setup_logger

import random

def slice_tweet(tweet: str, sentiment: str, min_prefix_words=2, min_completion_words=1) -> tuple[str, str] | None:
    """
    Slice the tweet into a prefix and completion, formatted with sentiment control.
    Returns (prompt, completion) tuple or None if tweet is too short.
    """
    words = tweet.strip().split()
    if len(words) < min_prefix_words + min_completion_words:
        return None

    min_cut = min_prefix_words
    max_cut = len(words) - min_completion_words
    cut_index = random.randint(min_cut, max_cut)

    prefix = " ".join(words[:cut_index])
    completion = " ".join(words[cut_index:])

    prompt = f"<sentiment: {sentiment}> {prefix}"
    return (prompt, completion)


def load_prompt_templates(template_path: Path) -> Dict[str, str]:
    if not template_path.is_file():
        raise ValueError(f"Template file not found: {template_path}")
    
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            templates = json.load(file)
        if "system_prompt" not in templates or "user_prompt" not in templates:
            raise ValueError("Template file must include 'system_prompt' and 'user_prompt' keys.")
        return templates
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding template file: {e}")

def formatter(
        data: dict,
        dataset_name: str,
        templates: Dict[str, str],
        is_evaluation: bool=False,
        logger: Optional[Logger]=None
        ) -> Dict[str, List[Dict[str, str]]]:
    try:
        tweet_id = data['textID']
        tweet = data['text']
        sentiment = data['sentiment']
    except KeyError as e:
        if logger:
            logger.error(f"KeyError encountered: {e}")
        raise KeyError(f"Missing key(s) while attempting to format dataset for training.")

    sliced = slice_tweet(tweet=tweet, sentiment=sentiment)
    if not sliced:
        return None
    prompt, completion = sliced
    system_prompt = templates["system_prompt"]
    user_prompt = templates["user_prompt"].format(prefix=prompt)

    if not is_evaluation:
        return {
            "dataset": dataset_name,
            "id": tweet_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": completion}
            ],
            # "judging": {"content": data['content'], "question": data['question']}
        }
    else:
        return {
            "dataset": dataset_name,
            "id": tweet_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # "judging": {"content": data['content'], "question": data['question']}
        }


def start(
        dataset_name: str,
        input_path: Path,
        template_path: Path,
        output_path: Optional[Path],
        is_evaluation: bool=False):
    
    if not input_path.is_file():
        raise ValueError(f'File not found: {input_path}. Either path is not a file or file does not exist..')
    
    # default output path if not specified
    if output_path is None:
        output_dir = Path(input_path).parent
        output_path = output_dir / f"prepared_{input_path.name}"
    
    logger: Logger = setup_logger(Path(__file__).stem + "_" + input_path.stem)

    # load prompt templates
    templates = load_prompt_templates(template_path)
    logger.info(f"Loaded prompt templates from {template_path}")

    # read raw train jsonl file
    with open(input_path, 'r', encoding='utf-8') as in_file:
        tweets_data = [json.loads(line) for line in in_file]

    res = []
    for data in tweets_data:
        formatted = formatter(
            data, 
            dataset_name, 
            templates, 
            is_evaluation,
            logger
        )
        if not formatted:
            continue
        res.append(formatted)

    write_jsonl_file(content=res, output_path=output_path)
    logger.info(f"Dataset of size {len(res)} prepared at {output_path.resolve()}")

if __name__ == '__main__':
    CLI(start, as_positional=False)

# python -m dataset_preparation.tweet_generation_preparer --dataset_name tweets_sentiment_generation --input_path datasets/raw_datasets/tweets/train.jsonl --template_path dataset_preparation/prompt_templates/tweet_sentiment_generation.json --output_path datasets/formatted_datasets/prepared_sentiment_generation.jsonl

# import json

# input_file = '../archive/train.jsonl'
# output_file = '../archive/prepared_train.jsonl'  

# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         record = json.loads(line)

#         record['original_text'] = record.pop('text')

#         # Create new 'text' field as per the sentiment + text format
#         record['text'] = f"<sentiment: {record['sentiment']}> {record['original_text']}"

#         # Write to new JSONL file
#         json.dump(record, outfile)
#         outfile.write('\n')

# print(f"Transformed JSONL saved as {output_file}")