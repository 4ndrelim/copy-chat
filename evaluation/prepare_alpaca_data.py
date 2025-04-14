import csv
import json
import argparse
import os

def prepare_data(input_csv_path, output_dir, model_completion_col, ref_completion_col, generator_name):
    os.makedirs(output_dir, exist_ok=True)

    model_output_file = os.path.join(output_dir, "model_outputs.json")
    ref_output_file = os.path.join(output_dir, "reference_outputs.json")

    model_data_list = []
    ref_data_list = []

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            # Check required columns
            required_cols = [model_completion_col, ref_completion_col, 'prefix', 'sentiment']
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}. Missing: {', '.join(missing_cols)}")

            for row in reader:
                prefix = row.get('prefix', '').strip()
                sentiment = row.get('sentiment', 'neutral').strip()
                model_completion = row.get(model_completion_col, '').strip()
                ref_completion = row.get(ref_completion_col, '').strip()

                # Handle potential '<space>' token if needed
                if model_completion.startswith("<space>"):
                    model_completion = " " + model_completion[len("<space>"):]
                if ref_completion.startswith("<space>"):
                    ref_completion = " " + ref_completion[len("<space>"):]

                instruction = f"Complete the tweet with {sentiment} sentiment, starting with: '{prefix}'"

                model_data = {
                    "instruction": instruction,
                    "output": model_completion,
                    "generator": generator_name
                }
                ref_data = {
                    "instruction": instruction,
                    "output": ref_completion,
                    "generator": generator_name
                }
                model_data_list.append(model_data)
                ref_data_list.append(ref_data)

        with open(model_output_file, 'w', encoding='utf-8') as model_outfile:
             json.dump(model_data_list, model_outfile, indent=2)

        with open(ref_output_file, 'w', encoding='utf-8') as ref_outfile:
             json.dump(ref_data_list, ref_outfile, indent=2)

        print(f"Data prepared successfully.")
        print(f"Model outputs: {model_output_file}")
        print(f"Reference outputs: {ref_output_file}")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except ValueError as ve:
        print(f"Error processing CSV: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CSV data for AlpacaEval.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", default="evaluation/alpaca_eval_prepared_data", help="Directory to save the JSON output files.")
    parser.add_argument("--model_completion_col", default="completion", help="Column name for the model's completion in the CSV.")
    parser.add_argument("--ref_completion_col", default="original_completion", help="Column name for the reference (original) completion in the CSV.")
    parser.add_argument("--generator_name", default="trump_v4_qwen_mimicry", help="Name of the generator.")

    args = parser.parse_args()

    prepare_data(args.input_csv, args.output_dir, args.model_completion_col, args.ref_completion_col, args.generator_name)
