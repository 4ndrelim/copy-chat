import json

input_file = '../archive/train.jsonl'
output_file = '../archive/prepared_train.jsonl'  

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        record = json.loads(line)

        record['original_text'] = record.pop('text')

        # Create new 'text' field as per the sentiment + text format
        record['text'] = f"<sentiment: {record['sentiment']}> {record['original_text']}"

        # Write to new JSONL file
        json.dump(record, outfile)
        outfile.write('\n')

print(f"Transformed JSONL saved as {output_file}")
