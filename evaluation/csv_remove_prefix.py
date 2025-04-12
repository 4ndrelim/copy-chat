import argparse
import os
import pandas as pd

def remove_prefix(original, prefix):
    if isinstance(original, str) and isinstance(prefix, str) and original.startswith(prefix):
        return original[len(prefix):]
    return original

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file, encoding="utf-8")

    # Create new column by removing prefix from original.
    df['orig_without_prefix'] = df.apply(
        lambda row: remove_prefix(row.get('original', ''), row.get('prefix', '')),
        axis=1
    )
    
    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Remove prefix from original text in a CSV file and save the output with a new column 'orig_without_prefix'."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output_file", help="Path to output CSV file; defaults to results_<input_file>", default=None)
    args = parser.parse_args()

    if args.output_file is None:
        input_filename = os.path.basename(args.input_file)
        args.output_file = f"results_{input_filename}"

    process_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
