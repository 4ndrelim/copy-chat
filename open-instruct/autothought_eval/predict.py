import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from dataset_preparation.dataset_preparer import load_prompt_templates


class VLLMInferenceClient:
    """
    A client to interact with a vLLM server for text generation.
    """
    def __init__(self, base_url: str, model: str, prompts_template_path: Path):
        self.client = OpenAI(
            base_url=base_url,
            api_key="None"
        )
        self.prompts_template = load_prompt_templates(prompts_template_path)
        self.model = model

    def generate(self, query_chunk_pair: dict) -> str:
        """
        Sends a request to the vLLM server to generate text based on the prompt.
        """
        query = query_chunk_pair['query']
        chunk = query_chunk_pair['chunk']

        system_prompt = self.prompts_template['system_prompt']
        user_prompt = self.prompts_template["user_prompt"].format(query=query, chunk=chunk)

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stop=["</s>"]
        )
        return chat_response.choices[0].message.content

def main(args):
    base_url = args.base_url
    prompts_template_path = Path(args.prompts_template_path)
    model = args.model
    client = VLLMInferenceClient(base_url, model, prompts_template_path)
    with open(args.input_file, "r", encoding='utf-8') as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    # process and store results
    results = []
    for query_chunk_pair in tqdm(data, desc="Processing chunks"):
        try:
            output: str = client.generate(query_chunk_pair)
            query_chunk_pair['prediction'] = output
            results.append(query_chunk_pair)
        except Exception as e:
            print(f"Error processing data point: {query_chunk_pair}... - {e}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for text chunks using vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Address where the vLLM server is running")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use")
    parser.add_argument("--prompts_template_path", type=str, required=True, help="Path to the prompt template file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text JSONL file containing paragraphs")
    parser.add_argument("--output_file", type=str, default="autothought_eval/results/output.jsonl", help="Path to save the output JSONL file")
    args = parser.parse_args()

    main(args)
