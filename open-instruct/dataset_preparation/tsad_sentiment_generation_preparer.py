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

discarded_data = []
MIN_COMPLETION_TOKENS = 3

def slice_tweet(
    text: str, percent=0.5, verbose=False, discard=False
) -> tuple[str, str]:
    """
    Slice the tweet into a prefix and completion
    """
    # Tokenize text into IDs, then split, and decode back to text
    # Best effort attempt to keep at least min_completion_tokens cut off in the tweet
    text = text.strip()
    tokens = tokenizer(text).input_ids
    num_tokens = len(tokens) - 1  # includes the invisible start token
    if discard and num_tokens < 2:
        return "", text

    # there should be at least 1 token to start from, and 1 token to complete
    # also re add the starting token here
    split_point = max(1, min(int(num_tokens * percent), num_tokens - MIN_COMPLETION_TOKENS)) + 1
    prefix_tokens = tokenizer.decode(tokens[:split_point], skip_special_tokens=True)
    rest = tokenizer.decode(tokens[split_point:], skip_special_tokens=True)
    if verbose:
        print(f"slice_tweet args: {percent=}, {MIN_COMPLETION_TOKENS=}")
        print(f"{text=}, {split_point=}, {len(tokens)=}")
        print(f"{tokenizer.tokenize(text)=}")
        print(f"{prefix_tokens=}")
        print(f"{rest=}")
        print()

    return prefix_tokens, rest


def formatter(
    data: dict,
    dataset_name: str,
    templates: Dict[str, str],
    is_evaluation: bool = False,
    replicate_sentiments: bool = False,
    logger: Optional[Logger] = None,
    verbose: bool = False,
    discard: bool = False,
) -> List[Dict[str, List[Dict[str, str]]]]:
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

    if replicate_sentiments:
        sentiments = ["positive", "negative", "neutral"]
    else:
        sentiments = [sentiment]

    res = []
    for sentiment in sentiments:
        prefix, completion = slice_tweet(tweet, verbose=verbose, discard=discard)
        # discard if tweet is too short
        if discard and not prefix:
            print(f"Discarding {tweet}, too short")
            discarded_data.append(tweet_id)
            continue
        system_prompt = templates["system_prompt"]
        user_prompt = templates["user_prompt"].format(prefix=prefix)

        prompt = {
            "dataset": dataset_name,
            "id": tweet_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<sentiment: {sentiment}>{user_prompt}"},
            ],
        }

        if not is_evaluation:
            prompt["messages"].append(
                {
                    "role": "assistant",
                    "content": completion,
                }
            )

        res.append(prompt)
    return res


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


def start(
    dataset_name: str,
    input_path: Path,
    template_path: Path,
    output_path: Optional[Path],
    is_evaluation: bool = False,
    replicate_sentiments: bool = False,
    print_samples: int = 0,
    discard: bool = False
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
        train_data = [json.loads(line) for line in in_file]

    res = []
    count = 0
    for data in train_data:
        formatted = formatter(
            data,
            dataset_name,
            templates,
            is_evaluation,
            replicate_sentiments,
            logger,
            count < print_samples,
            discard
        )
        count += 1
        res.extend(formatted)

    print(f"{discarded_data=}")
    write_jsonl_file(content=res, output_path=output_path)
    logger.info(f"Dataset of size {len(res)} prepared at {output_path.resolve()}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        f"{os.environ['HOME']}/copy-chat/models/meta-llama-Llama-3.1-8B-Instruct",
        trust_remote_code=False,
        use_fast=False,
    )

    CLI(start, as_positional=False)

