"""
This script converts existing dataset to a format that can be parsed by allenAI Instruct
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Generator
import random


def random_binary_generator(seed: int, choices: List[int]=[1,2]) -> Generator[int, None, None]:
    if seed:
        random.seed(seed)
    while True:
        yield random.choice(choices)

def load_jsonl_file(file_path: Path) -> List[dict]:
    # jsonl useful for streaming large files since it doesnt require loading the 
    # whole file in memory like json format which stores multiple objects in an array
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            dict_obj = json.loads(line)
            data.append(dict_obj)

    assert len(data) > 0, f'{file_path.name} is empty!'
    return data

def load_jsonl_files(directory: Path, empty_ok: bool = False) -> List[dict]:
    """
    Recursively load all JSONL files within the directory and return a list of dictionaries
    """
    all_dicts = []
    jsonl_files = glob.glob(os.path.join(directory, '**', '*.jsonl'), recursive=True)
    if not empty_ok:
        assert len(jsonl_files) > 0, f"'{directory}' is empty or no jsonl files identified!"
    
    for file in jsonl_files:
        all_dicts.extend(load_jsonl_file(Path(file)))

    return sorted(all_dicts, key=lambda x: x['filename']) # just for replicability

def get_jsonl_paths(directory: Path) -> List[Path]:
    """
    Recursively searches for all JSONL paths from a directory.
    """
    jsonl_files = glob.glob(os.path.join(directory, '**', '*.jsonl'), recursive=True)
    jsonl_files.sort()
    return [Path(file).expanduser().resolve() for file in jsonl_files]

def write_jsonl_file(content: List[dict], output_path: Path):
    # create all ancestors directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in content:
            f.write(json.dumps(entry) + '\n')