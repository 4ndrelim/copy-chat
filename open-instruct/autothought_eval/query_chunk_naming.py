import argparse
import json
import re
from typing import List
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import csv

from dataset_preparation.dataset_preparer import load_prompt_templates


class VLLMInferenceClient:
    """
    A client to interact with a vLLM server for text generation.
    """
    def __init__(self, base_url: str, model:str, prompts_template_path: Path):
        self.client = OpenAI(
            base_url=base_url, # Mixtral
            api_key="None",
        )
        self.prompts_template = load_prompt_templates(prompts_template_path)
        self.model = model

    def generate(self, query_chunk_pair: dict) -> str:
        """
        Sends a request to the vLLM server to generate text based on the prompt.
        """
        query = query_chunk_pair['query'].replace('\n', '\n\t')
        chunk = query_chunk_pair['chunk'].replace('\n', '\n\t')
        prediction = query_chunk_pair['prediction'].replace('\n', '\n\t')

        system_prompt = "You are an expert in identifying the relevance between a given context and a generated title for the context."
        user_prompt = self.prompts_template["user_prompt"].format(query=query, chunk=chunk, prediction=prediction)

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stop=["</s>"],
            temperature=0,
            max_tokens=1000
        )
        return chat_response.choices[0].message.content

def clean_prediction(raw_prediction: str) -> dict:
    """
    Extracts the raw prediction dictionary.
    Returns the dictionary if valid, otherwise returns None.
    """
    try:
        prediction_dict = json.loads(raw_prediction)

        if isinstance(prediction_dict, dict) and 'explanation' in prediction_dict and 'suggestedTitle' in prediction_dict and 'score' in prediction_dict:
            return prediction_dict
        else:
            print('UNDETERMINED', raw_prediction)
            return None
        
    except Exception as e:
        print('UNDETERMINED', raw_prediction)
        print(e)
        return None


def compute_results(tp: int, fp: int, fn: int, tn: int) -> List[int]:
    acc = (tp+tn) / (tp+tn+fp+fn)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    return [acc, sn, sp]

def main(args):
    base_url = args.base_url
    prompts_template_path = Path(args.prompts_template_path)
    model = args.model
    client = VLLMInferenceClient(base_url, model, prompts_template_path)
    with open(args.input_file, "r", encoding='utf-8') as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    undetermined_count = 0

    scores = []
    results = []

    for query_chunk_pair in tqdm(data, desc="Processing chunks"):
        try:
            prediction = client.generate(query_chunk_pair)
            parsed = clean_prediction(prediction)

            results.append({"evaluation": parsed, "title": query_chunk_pair["prediction"], "user": query_chunk_pair['query'], "system": query_chunk_pair['chunk']})
            
            if parsed == None:
                undetermined_count += 1
                continue

            score = parsed['score']

            if score < 1 or score > 5:
                print(f"Invalid score: {score}")
                continue

            scores.append(score)

        except Exception as e:
            print(f"Error processing data point: {query_chunk_pair}... - {e}")

    print("\n=== Score Statistics ===")
    print(f"Number of scores: {len(scores)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}")
    print(f"Min score: {min(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Undetermined count: {undetermined_count} / {len(data)}")

    # Save results to JSONL file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for text chunks using vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Address where the vLLM server is running")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use")
    parser.add_argument("--prompts_template_path", type=str, required=True, help="Path to the prompt template file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text JSONL file containing paragraphs")
    parser.add_argument("--output_file", type=str, default="autothought_eval/results/output.jsonl", help="Path to save the output JSONL file")
    args = parser.parse_args()

    main(args)
