import argparse
import json
import re
from typing import List
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from dataset_preparation.dataset_preparer import load_prompt_templates


class VLLMInferenceClient:
    """
    A client to interact with a vLLM server for text generation.
    """
    def __init__(self, base_url: str, model:str, prompts_template_path: Path):
        self.client = OpenAI(
            base_url=base_url,
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

        system_prompt = self.prompts_template['system_prompt']
        user_prompt = self.prompts_template["user_prompt"].format(query=query, chunk=chunk)

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stop=["</s>"],
            temperature=0,
            max_tokens=5
        )
        return chat_response.choices[0].message.content

def clean_prediction(raw_prediction: str) -> str:
    """
    Tries to find 'relevant' or 'irrelevant' in the raw_prediction string.
    If neither or both are found, returns "undetermined".
    """
    # define search patterns
    relevant_pattern = re.compile(r'^relevant\b', re.IGNORECASE)
    irrelevant_pattern = re.compile(r'^irrelevant\b', re.IGNORECASE)

    # check occurrences of relevant and irrelevant
    relevant_found = bool(relevant_pattern.search(raw_prediction))
    irrelevant_found = bool(irrelevant_pattern.search(raw_prediction))
    
    # determine result based on findings
    if relevant_found and not irrelevant_found:
        return "relevant"
    elif irrelevant_found and not relevant_found:
        return "irrelevant"
    else:
        print('UNDETERMINED', raw_prediction)
        return "undetermined"


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

    # for calcuation of accuracy, sensitivity, specificity
    tp, fp, fn, tn = 0, 0, 0, 0
    undetermined_count = 0

    for query_chunk_pair in tqdm(data, desc="Processing chunks"):
        try:
            label = query_chunk_pair['answer']
            prediction: str = clean_prediction(client.generate(query_chunk_pair))
            if prediction == "undetermined":
                undetermined_count += 1
            else:
                if prediction == label == 'relevant':
                    tp += 1
                elif prediction == label == 'irrelevant':
                    tn += 1
                elif prediction == 'relevant' != label:
                    fp += 1
                elif prediction == 'irrelevant' != label:
                    fn += 1
                else:
                    assert False, "Should not happen during update of confusion matrix!"

        except Exception as e:
            print(f"Error processing data point: {query_chunk_pair}... - {e}")

    print("\n=== Confusion Matrix ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    accuracy, sensitivity, specificity = compute_results(tp, fp, fn, tn)
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy   : {accuracy:.2%}")
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Undetermined count: {undetermined_count} / {len(data)}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for text chunks using vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Address where the vLLM server is running")
    parser.add_argument("--model", type=str, required=True, help="Specify the model to use")
    parser.add_argument("--prompts_template_path", type=str, required=True, help="Path to the prompt template file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text JSONL file containing paragraphs")
    parser.add_argument("--output_file", type=str, default="autothought_eval/results/output.jsonl", help="Path to save the output JSONL file")
    args = parser.parse_args()

    main(args)
