"""
This script converts existing datasets to a format that can be parsed by allenAI Instruct
Usage:
python -m dataset_preparation.tweet_generation_preparer --dataset_name tweets_sentiment_generation --input_path datasets/raw_datasets/tweets/train.jsonl --template_path dataset_preparation/prompt_templates/tweet_sentiment_generation.json --output_path datasets/formatted_datasets/prepared_sentiment_generation.jsonl
"""

from pathlib import Path
from typing import List, Dict, Optional
from logging import Logger
import json
import sys
import os
from jsonargparse import CLI
from transformers import AutoTokenizer

# force add current directory of script to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from dataset_preparation.dataset_utils import (
    write_jsonl_file,
)

from utils.logger import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(
    f"{os.environ['HOME']}/copy-chat/models/meta-llama-Llama-3.1-8B-Instruct",
    trust_remote_code=False,
    use_fast=False,
)


def split_text(
    text, percent=0.5, min_prefix_tokens=7, min_completion_tokens=1, verbose=False
):
    # Tokenize text into IDs, then split, and decode back to text
    # Prioritise having completion tokens over prefix tokens
    # This is to ensure that the model has enough context to generate a response
    if verbose:
        print(
            f"Splitting {text=} with {percent=}, {min_prefix_tokens=}, {min_completion_tokens=}"
        )
    tokens = tokenizer(text).input_ids
    split_point = int(len(tokens) * percent)
    if len(tokens) < min_prefix_tokens + min_completion_tokens:
        split_point = len(tokens) - 1
    elif len(tokens) - split_point < min_completion_tokens:
        split_point = len(tokens) - min_completion_tokens
    elif split_point < min_prefix_tokens:
        split_point = min_prefix_tokens

    prefix_tokens = tokens[:split_point]
    rest = tokens[split_point:]
    return tokenizer.decode(prefix_tokens, skip_special_tokens=True), tokenizer.decode(
        rest, skip_special_tokens=True
    )


def slice_tweet(tweet: str, sentiment: str):
    # -> tuple[str, str] | None
    """
    Slice the tweet into a prefix and completion, formatted with sentiment control.
    Returns (prompt, completion) tuple or None if tweet is too short.
    """
    prefix, completion = split_text(
        text=tweet,
        percent=0.5,
    )

    prompt = f"<sentiment: {sentiment}> {prefix}"
    return (prompt, completion)


def load_prompt_templates(template_path: Path) -> Dict[str, str]:
    if not template_path.is_file():
        raise ValueError(f"Template file not found: {template_path}")

    try:
        with open(template_path, "r", encoding="utf-8") as file:
            templates = json.load(file)
        if "system_prompt" not in templates or "user_prompt" not in templates:
            raise ValueError(
                "Template file must include 'system_prompt' and 'user_prompt' keys."
            )
        return templates
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding template file: {e}")


def formatter(
    data: dict,
    dataset_name: str,
    templates: Dict[str, str],
    is_evaluation: bool = False,
    logger: Optional[Logger] = None,
) -> Dict[str, List[Dict[str, str]]]:
    try:
        tweet_id = data["textID"]
        tweet = data["text"]
        sentiment = data["sentiment"]
    except KeyError as e:
        if logger:
            logger.error(f"KeyError encountered: {e}")
        raise KeyError(
            f"Missing key(s) while attempting to format dataset for training."
        )

    sliced = slice_tweet(tweet=tweet, sentiment=sentiment)
    if not sliced:
        return None
    prompt, completion = sliced
    system_prompt = templates["system_prompt"]
    user_prompt = templates["user_prompt"].format(prefix=prompt)

    inter = {
        "dataset": dataset_name,
        "id": tweet_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if not is_evaluation:
        inter["messages"].append(
            {
                "role": "assistant",
                "content": completion,
            }
        )
    return inter


def start(
    dataset_name: str,
    input_path: Path,
    template_path: Path,
    output_path: Optional[Path],
    is_evaluation: bool = False,
):

    if not input_path.is_file():
        raise ValueError(
            f"File not found: {input_path}. Either path is not a file or file does not exist.."
        )

    # default output path if not specified
    if output_path is None:
        output_dir = Path(input_path).parent
        output_path = output_dir / f"prepared_{input_path.name}"

    logger: Logger = setup_logger(Path(__file__).stem + "_" + input_path.stem)

    # load prompt templates
    templates = load_prompt_templates(template_path)
    logger.info(f"Loaded prompt templates from {template_path}")

    # read raw train jsonl file
    with open(input_path, "r", encoding="utf-8") as in_file:
        # # print debug
        # tweets_data = []
        # for line in in_file:
        #     print(line)
        #     print(json.loads(line))
        #     tweets_data.append(json.loads(line))
        tweets_data = [json.loads(line) for line in in_file]

    res = []
    for data in tweets_data:
        formatted = formatter(data, dataset_name, templates, is_evaluation, logger)
        if not formatted:
            continue
        res.append(formatted)

    write_jsonl_file(content=res, output_path=output_path)
    logger.info(f"Dataset of size {len(res)} prepared at {output_path.resolve()}")


if __name__ == "__main__":
    CLI(start, as_positional=False)
