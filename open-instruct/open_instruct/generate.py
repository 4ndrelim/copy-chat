import argparse
import json
from tqdm import tqdm
from openai import OpenAI

class VLLMInferenceClient:
    """
    A client to interact with a vLLM server for text generation.
    """
    def __init__(self, base_url):
        self.base_url = base_url
        
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{self.base_url}/v1",
        )

    def generate(self, prompt):
        """
        Sends a request to the vLLM server to generate text based on the prompt.
        """

        chat_response = self.client.chat.completions.create(
            model="./",
            messages=[
                {"role": "system", "content": "You are an expert in identifying the relevance of the chunk and query. The chunk is relevant when it is semantically useful for the query."},
                {"role": "user", "content": prompt},
            ],
            stop=["</s>"]
        )
        print(chat_response.choices[0].message.content)
        exit()
        return chat_response.choices[0].message.content

def generate_question_for_chunk(client, query_chunk_pair: dict):
    """
    Generates a question for a given query chunk pair using the vLLM server.
    """
    query = query_chunk_pair['query']
    chunk = query_chunk_pair['chunk']
    prompt = (
        f"Query: {query}\n\n"
        f"Chunk: {chunk}"
    )
    return client.generate(prompt)

def main(args):
    client = VLLMInferenceClient(f"http://localhost:{args.port}")

    with open(args.input_file, "r", encoding='utf-8') as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    # Process each chunk and generate questions
    results = []
    for query_chunk_pair in tqdm(data, desc="Processing chunks"):
        try:
            output: str = generate_question_for_chunk(client, query_chunk_pair)
            query_chunk_pair['prediction'] = output
            results.append(query_chunk_pair)
        except Exception as e:
            print(f"Error processing data point: {query_chunk_pair}... - {e}")

    with open(args.output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions for text chunks using vLLM")
    parser.add_argument("--port", type=int, default=8000, help="Port where the vLLM server is running")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file containing paragraphs")
    parser.add_argument("--output_file", type=str, default="data/output.jsonl", help="Path to save the output JSONL file")
    args = parser.parse_args()

    main(args)
