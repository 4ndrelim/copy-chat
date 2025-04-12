import argparse
import os

def convert(input_file, output_file):
    with open(input_file, 'r', encoding='latin-1') as infile:
        text = infile.read()
        
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(text)

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV file from Latin-1 to UTF-8 encoding."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    args = parser.parse_args()

    # Extract the base filename and extension.
    input_filename = os.path.basename(args.input_file)
    file_root, file_ext = os.path.splitext(input_filename)
    output_filename = f"{file_root}_conv{file_ext}"
    
    # Save the output in the same directory as the input file.
    input_directory = os.path.dirname(args.input_file)
    output_path = os.path.join(input_directory, output_filename)
    
    convert(args.input_file, output_path)
    print(f"Conversion complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
