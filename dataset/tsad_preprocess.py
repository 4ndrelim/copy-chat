"""
This script takes in dataset and gives you the option to:
- Do preprocessing
- Add end tokens

Feel free to include additional preprocessing steps.

Run with `python tsad_preprocess.py <csv filepath> [--preprocess] [--endtoken]`
May need to run `python -m spacy download en_core_web_sm`
"""

import pandas as pd
import re
import spacy
import os
import argparse
from tqdm import tqdm

print("loading spacy model...")
nlp = spacy.load("en_core_web_sm")
print("spacy model loaded")


def main(filepath, preprocess, endtoken, encoding):
    tqdm.pandas()
    output_filename = os.path.basename(filepath)[:-4]

    df = pd.read_csv(filepath, encoding=encoding)

    if preprocess:
        ### FILTERING ###
        df = apply_preprocess_filtering(df, is_useful_len)
        df = apply_preprocess_filtering(df, is_grammatically_complete)

        ### OTHER KINDS OF PREPROCESSING ###
        mask = df["text"].progress_apply(has_non_english_keyboard_chars)
        count = mask.sum()
        print(f"No. of rows with non-english keyboard characters: {count}")
        # print(f"The rows:")
        # df[mask].apply(print)
        df = df[~mask].reset_index(drop=True)

        df["text"] = (
            df["text"].progress_apply(replace_tagging).progress_apply(replace_link)
        )

        output_filename += "_preprocessed"

    if endtoken:
        df["text"] = df["text"].astype(str) + "|<end>|"

        output_filename += "_endtoken"

    output_filename += ".csv"
    df.to_csv(output_filename, index=False)
    print(f"{output_filename} file created")


def apply_preprocess_filtering(df: pd.DataFrame, func):
    # func parameter is a function that returns True or False
    mask = df["text"].progress_apply(func)
    dropped = (~mask).sum()
    print(f"No. of rows {func.__name__} filter dropped: {dropped}")
    return df[mask].reset_index(drop=True)


def is_useful_len(text):
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"(https?://[^\s]+|www\.[^\s]+)", "", text)
    text = text.strip()
    if text.lower().startswith("rt"):
        text = text[2:].strip()
    words = text.split()
    return len(words) > 3


def is_grammatically_complete(text):
    text = str(text).strip()

    doc = nlp(text)

    has_subject = False
    has_verb = False

    for token in doc:
        # Accept common subject types
        if token.dep_ in (
            "nsubj",
            "nsubjpass",
            "csubj",
            "csubjpass",
            "expl",
            "agent",
            "compound",
        ):
            has_subject = True

        # Accept main verbs and auxiliary verbs ("is", "was", "be", etc.)
        if token.pos_ in ("VERB", "AUX"):
            has_verb = True

    # Also allow imperative sentences (commands) which may have no explicit subject
    if has_verb and not has_subject:
        # If it starts with a verb, likely an imperative sentence ("Go outside", "Check this out")
        first_token = doc[0]
        if first_token.pos_ in ("VERB", "AUX"):
            return True

    return has_subject and has_verb


allowed_pattern = r"^[\s\x21-\x7E]*$"


def has_non_english_keyboard_chars(text):
    return not bool(re.match(allowed_pattern, str(text)))


def replace_tagging(text):
    # Replaces all taggings with <username>, a standardized placeholder to represent all usernames
    pattern = r"@(?:\w+|(?:\s+\w{6,}))"
    text = re.sub(pattern, "{username}", text)
    return text.strip()


def replace_link(text):
    # Replaces all links with <link>, a standardized placeholder to represent all links
    pattern = r"(https?://[^\s]+|www\.[^\s]+)"
    text = re.sub(pattern, "{link}", text)
    return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess mimicry dataset")
    parser.add_argument("filepath", type=str, help="Path to csv file")
    parser.add_argument(
        "--preprocess", action="store_true", help="Preprocess/Clean dataset"
    )
    parser.add_argument("--endtoken", action="store_true", help="Add end token to text")
    parser.add_argument(
        "--encoding", type=str, default="utf-8", help="Encoding of the CSV file"
    )
    args = parser.parse_args()

    main(args.filepath, args.preprocess, args.endtoken, args.encoding)
