# Formats output of sentiment generation (phase 2) instruct models. Writes a csv with original text, input, and output
# Paths can be relative (uses `root_folder` as the base path)
# e.g. use (for match_v08): python predictions/sentiment_generation/prettify.py -r dataset/tweet_sentiment/test/test.jsonl -f open-instruct/datasets/formatted_datasets/tsad_08_test.jsonl -m predictions/sentiment_generation/tsad_llama_v08.jsonl

import collections
import argparse, csv, json, os, sys, re
from pathlib import Path

# from transformers import AutoTokenizer

csv_filename = "match.csv"
root_folder = Path(__file__).parent.parent.parent
error_count = {"total": 0, "strip": 0, "insert": 0, "discard": 0}


def write_csv(hashmap: dict):
    with open(root_folder / csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["textID", "original", "sentiment", "prefix", "completion", "combined"]
        )
        for key, value in hashmap.items():
            writer.writerow(
                [
                    key,
                    value["original"],
                    value["sentiment"],
                    value["prefix"],
                    value["completion"],
                    value["combined"],
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--raw_test_file",
        type=str,
        required=True,
        help="(relative) path to the raw test file.",
    )
    parser.add_argument(
        "-f",
        "--formatted_dataset_file",
        type=str,
        required=True,
        help="(relative) path to the formatted dataset file.",
    )
    parser.add_argument(
        "-m",
        "--model_output_file",
        type=str,
        required=True,
        help="(relative) path to the model output file.",
    )
    return parser.parse_args()


def edit_distance(word1: str, word2: str) -> int:
    dp = [[0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        dp[i][0] = i
    for j in range(len(word2) + 1):
        dp[0][j] = j
    for i in range(len(word1)):
        for j in range(len(word2)):
            if word1[i] == word2[j]:
                dp[i + 1][j + 1] = dp[i][j]
            else:
                dp[i + 1][j + 1] = min(dp[i][j], dp[i][j + 1], dp[i + 1][j]) + 1
    return dp[-1][-1]


def match_v08(raw_test_file_path, model_output_file_path, output_file_path):
    # e.g. use: python prettify.py -r dataset/tweet_sentiment/test/test.jsonl -f open-instruct/datasets/formatted_datasets/tsad_08_test.jsonl -m predictions/sentiment_generation/tsad_llama_v08.jsonl
    with open(root_folder / raw_test_file_path) as o, open(
        root_folder / model_output_file_path
    ) as f, open(root_folder / output_file_path) as v:
        hashmap = collections.defaultdict(dict)
        for line in o.readlines():
            txt = json.loads(line)
            hashmap[txt["textID"]]["original"] = txt["text"]
            hashmap[txt["textID"]]["sentiment"] = txt["sentiment"]
        for line in f.readlines():
            txt = json.loads(line)
            hashmap[txt["id"]]["prefix"] = txt["messages"][1]["content"]
        for line in v.readlines():
            txt = json.loads(line)
            assert hashmap[txt["id"]]["prefix"] == txt["messages"][1]["content"]
            if txt["output"].startswith("<space>"):
                # remove <space> tag
                # print(
                #     f"removing space for {txt['id']}, |{txt['output']}|. result: |{' ' + txt['output'][7:]}|"
                # )
                txt["output"] = " " + txt["output"][7:]
                # check if there are any wrongly formatted <space> tokens
                # if "space" in txt["output"]:
                #     print(f"found space in output for {txt['id']}, |{txt['output']}|")
            hashmap[txt["id"]]["completion"] = txt["output"]
            hashmap[txt["id"]]["combined"] = (
                hashmap[txt["id"]]["prefix"] + txt["output"]
            )
    write_csv(hashmap)


def match_v06(raw_test_file_path, model_output_file_path, output_file_path):
    # e.g. use: python prettify.py -r dataset/tweet_sentiment/test/test.jsonl -f open-instruct/datasets/formatted_datasets/tsad_06_test.jsonl -o predictions/sentiment_generation/tsad_llama_v06.jsonl
    with open(root_folder / raw_test_file_path) as o, open(
        root_folder / model_output_file_path
    ) as f, open(root_folder / output_file_path) as v:
        hashmap = collections.defaultdict(dict)
        for line in o.readlines():
            txt = json.loads(line)
            hashmap[txt["textID"]]["original"] = txt["text"]
            hashmap[txt["textID"]]["sentiment"] = txt["sentiment"]
        for line in f.readlines():
            txt = json.loads(line)
            hashmap[txt["id"]]["prefix"] = txt["messages"][1]["content"]

        global error_count
        sen_regex = re.compile(r"<sentiment: (positive|negative|neutral)>")
        for line in v.readlines():
            txt = json.loads(line)
            hashmap[txt["id"]]["combined"] = txt["output"]

            # remove sentiment tag
            if sen_regex.match(txt["messages"][1]["content"]) is None:
                raise ValueError(
                    f"Sentiment tag not found in user content: {txt['messages'][1]['content']}"
                )
            prefix = sen_regex.sub("", txt["messages"][1]["content"])
            # remove prefix from completion
            if not txt["output"].startswith(prefix):
                # raise ValueError(
                #     f"Prefix not found in output: {txt['id']}, {txt['output']} vs {prefix}"
                # )
                print(f"error {txt['id']=}")
                print(
                    f"Prefix not found in output: ||{txt['output']}|| vs ||{prefix}||"
                )
                print()
                error_count["total"] += 1
                # try a few strategies
                # 1. strip prefix
                if txt["output"].startswith(prefix.strip()):
                    error_count["strip"] += 1
                    print(
                        f"Stripping prefix: {txt['id']}\n|{txt['output']}|\n|{prefix}|\n"
                    )
                    prefix = prefix.strip()
                else:
                    # 2. insert a space somewhere in output
                    for i in range(0, len(txt["output"])):
                        if txt["output"].startswith(prefix[:i] + " " + prefix[i:]):
                            error_count["insert"] += 1
                            print(
                                f"Inserting space in prefix: {txt['id']}\n|{txt['output']}|\n|{prefix}|\n"
                            )
                            prefix = prefix[:i] + " " + prefix[i:]
                            break
                    else:
                        # check which prefix has the least edit distance
                        dists = [
                            edit_distance(prefix, txt["output"][:i])
                            for i in range(len(txt["output"]))
                        ]
                        min_dist = min(range(len(dists)), key=dists.__getitem__)
                        print(
                            f"Discarding: (min edit distance [{min_dist}]={dists[min_dist]})"
                        )
                        print(f"input:   |{prefix}")
                        print(f"output:  |{txt['output']}")
                        print(f'original:|{hashmap[txt["id"]]["original"]}')
                        print()
                        hashmap[txt["id"]]["completion"] = "<error:discard>"
                        error_count["discard"] += 1
                        continue
            hashmap[txt["id"]]["completion"] = txt["output"][len(prefix) :]
            # print(f'{hashmap[txt["id"]]["combined"]=}')
            # print(f'{hashmap[txt["id"]]["completion"]=}')
            # print(f'{hashmap[txt["id"]]=}')
            # print(f"{prefix=}")
            # print(f"{txt=}")
            # return

    write_csv(hashmap)


if __name__ == "__main__":
    # error_count = 0
    args = parse_args()
    # match_v08(args.raw_test_file, args.formatted_dataset_file, args.model_output_file)
    match_v06(args.raw_test_file, args.formatted_dataset_file, args.model_output_file)
    print(f"{error_count=}")
