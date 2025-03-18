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


# question as user prompt
# chunk as response
def formatter(
        data: dict,
        dataset_name: str,
        templates: Dict[str, str],
        is_evaluation: bool=False,
        logger: Optional[Logger]=None
        ) -> Dict[str, List[Dict[str, str]]]:
    try:
        tweet_id = data['id']
        tweet = data['tweet']
    except KeyError as e:
        if logger:
            logger.error(f"KeyError encountered: {e}")
        raise KeyError(f"Missing key(s) while attempting to format dataset for training.")

    system_prompt = templates["system_prompt"]
    user_prompt = templates["user_prompt"].format(tweet=tweet)

    if not is_evaluation:
        sentiment = data['sentiment']
        rationale = data['rationale']
        answer = f"Reason: {rationale}\nSentiment: {sentiment}"
        return {
            "dataset": dataset_name,
            "id": tweet_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer}
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

    # read csv
    with open(input_path, mode='r', encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f, delimiter=',')

    tweets_data = [
        {
            'id': row['textID'],
            'tweet': row['text'],
            'rationale': row['selected_text'] if 'selected_text' in row else None,
            'sentiment': row['sentiment'] if 'sentiment' in row else None
        }
        for _, row in df.iterrows()
    ]
    res = []
    for data in tweets_data:
        res.append(formatter(
            data, 
            dataset_name, 
            templates, 
            is_evaluation,
            logger
            )
        )

    write_jsonl_file(content=res, output_path=output_path)
    logger.info(f"Dataset of size {len(res)} prepared at {output_path.resolve()}")

if __name__ == '__main__':
    CLI(start, as_positional=False)
