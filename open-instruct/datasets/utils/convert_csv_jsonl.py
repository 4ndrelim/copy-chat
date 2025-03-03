import csv
import json
import random

# Input CSV file path
csv_file_path = "./books/BooksDatasetClean.csv"
# Output JSONL file path
jsonl_file_path = "./books_data.jsonl"

# Variations of user queries
query_variations = [
    "Tell me about {title}",
    "Give me details on {title}",
    "What is {title} about?",
    "I need info on {title}",
    "Can you describe the book {title}?",
    "Provide information on {title}",
    "Share details about {title}",
    "What can you tell me about {title}?"
]

# Read the CSV and convert it to JSONL with input-output format
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file, \
     open(jsonl_file_path, mode="w", encoding="utf-8") as jsonl_file:
    
    reader = csv.DictReader(csv_file)  # Read CSV as dictionary
    limit = 10000
    for row in reader:
        if limit == 0:
            break
        # Replace empty strings or null values with <NOT_FOUND>
        row = {key: (value if value and value.strip() else "<NOT_FOUND>") for key, value in row.items()}
        
        # Select a random query variation
        input_query = random.choice(query_variations).format(title=row['Title'])
        
        output_response = (
            f"**Title**: {row['Title']}\n"
            f"**Category**: {row['Category']}\n"
            f"**Publisher**: {row['Publisher']}\n"
            f"**Est. Price**: {row['Price Starting With ($)']}\n"
            f"**Publish Year**: {row['Publish Date (Year)']}\n"
            f"**About**:\n{row['Description']}\n"
        )
        
        jsonl_file.write(json.dumps({"input": input_query, "output": output_response}) + "\n")
        limit -= 1
print(f"Conversion completed. JSONL file saved as {jsonl_file_path}")
