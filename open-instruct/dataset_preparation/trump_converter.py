"""
This script is for converting the Trump dataset on kaggle for finetuning.
https://www.kaggle.com/datasets/codebreaker619/donald-trump-tweets-dataset
"""
import csv
import json

def start(input_csv: str, output_json: str):
    data = []
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if row.get("id") and row.get("text"):
                data.append({
                    "id": row["id"],
                    "text": row["text"]
                })

    with open(output_json, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} tweets to {output_json}")

if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(start, as_positional=False)
