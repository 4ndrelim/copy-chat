import pandas as pd
import json

# Load CSV
df = pd.read_csv('./tweet_sentiment/train/train_vanilla_augment.csv', encoding='latin1')

# Select required fields
fields_to_keep = ['textID', 'text', 'selected_text', 'sentiment']
df_filtered = df[fields_to_keep].dropna()

# Convert to JSONL
output_file = './tweet_sentiment/train/train_vanilla_augment.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for record in df_filtered.to_dict(orient='records'):
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')

print(f"JSONL file saved as {output_file}")
