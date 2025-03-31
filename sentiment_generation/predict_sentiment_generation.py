import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import re
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor


class VLLMInferenceClient:
    """
    A client to interact with a vLLM server for text generation.
    """
    def __init__(self, base_url: str, model: str, prompt_template: dict, temperature: float):
        self.client = OpenAI(
            base_url=base_url,
            api_key="None"
        )
        self.system_prompt = prompt_template['system_prompt']
        self.user_prompt = prompt_template['user_prompt']
        self.model = model
        self.temp = temperature

    def generate(self, prefix: str, sentiment: str) -> str:
        """
        Sends a request to the vLLM server for generation.
        """
        system_prompt = self.system_prompt
        prefix_with_sentiment = f"<sentiment: {sentiment}>{prefix}"
        user_prompt = self.user_prompt.format(prefix=prefix_with_sentiment)

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temp,
            stop=["</s>"]
        )
        return chat_response.choices[0].message.content


def try_parse(response: str):
    return response


def write_to_csv(results: List[dict], output_path: str):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results successfully written to {output_path}")


async def process_row(index, row, client, executor, res, total):
    loop = asyncio.get_running_loop()
    text_id = row['textID']
    sentiment = row['sentiment']
    original_text = row['original_text']
    prefix_text = row['prefix_text']
    raw_output = await loop.run_in_executor(executor, client.generate, prefix_text, sentiment)
    completion = prefix_text + " " + try_parse(raw_output)
    res.append(
        {
            "textID": text_id,
            "sentiment": sentiment,
            "original_text": original_text,
            "prefix_text": prefix_text,
            "predicted": raw_output,
            "combined": completion
        }
    )
    print(f"{index+1}/{total} completed..")


async def main(args):
    base_url = args.base_url
    input_csv = args.input_file
    output_csv = args.output_file
    model = args.model
    temp = args.temperature
    prompt_template_path = args.prompt_template
    with open(prompt_template_path, "r", encoding="utf-8") as file:
        prompt_template = json.load(file)
    client = VLLMInferenceClient(base_url, model, prompt_template, temp)

    df = pd.read_csv(input_csv, encoding='latin1')
    res = []
    total = len(df)

    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            process_row(index, row, client, executor, res, total)
            for index, row in df.iterrows()
        ]
        await asyncio.gather(*tasks)

    write_to_csv(res, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for text chunks using vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8282/v1", help="Address where the vLLM server is running")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--prompt_template", type=str, required=True, help="Path to prompt template to apply.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, default="autothought_eval/results/output.jsonl", help="Path to save the output JSONL file")
    args = parser.parse_args()

    asyncio.run(main(args))

# qwen
# python predict_sentiment_generation.py --base_url http://localhost:8282/v1 --model my_qwen_model --prompt_template ./open-instruct/dataset_preparation/prompt_templates/tweet_sentiment_generation.json --input_file sentiment_generation_test.csv --output_file qwen_sen_gen_prediction.csv
