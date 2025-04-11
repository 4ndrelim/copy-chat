import argparse
import os
import pandas as pd
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)

def compute_nli(premise, hypothesis):
    """
    Computes NLI metrics for given premise and hypothesis with 'roberta-large-mnli'.
    Returns:
        tuple: (nli_label, confidence_score)
    """
    # Encode premise, hypothesis pair
    inputs = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True)
    outputs = model(inputs)[0]  # logits
    probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
    # roberta-large-mnli's labels:
    labels = ["contradiction", "neutral", "entailment"]
    max_idx = probabilities.argmax()
    label = labels[max_idx]
    score = probabilities[max_idx]
    return label, score

def compute_sentence_order_score(completion, reference):
    """
    Computes simple sentence ordering score between the completion and reference.
    
    It sentence-tokenises each text. Then, for sentences in the completion appearing exactly 
    in the reference, it calculates the proportion of sentence pairs in the correct order.
    
    Returns:
        float or None: The ratio of correctly ordered pairs (0 to 1) or None if fewer than 2 sentences match.
    """
    cand_sentences = nltk.sent_tokenize(completion)
    ref_sentences = nltk.sent_tokenize(reference)
    
    positions = []
    for sent in cand_sentences:
        try:
            pos = ref_sentences.index(sent)
            positions.append(pos)
        except ValueError:
            # If sentence doesn't match exactly, skip
            continue

    if len(positions) < 2:
        return None

    total_pairs = 0
    correct_pairs = 0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            total_pairs += 1
            if positions[i] < positions[j]:
                correct_pairs += 1
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def evaluate_rows(input_file, metric="nli"):
    """
    Reads the CSV and calls NLI or sentence ordering coherence evaluation functions.
    
    Returns an updated DataFrame with columns: nli_label, nli_score, sentence_order_score.
    """
    df = pd.read_csv(input_file, encoding="utf-8")
    
    if metric.lower() == "nli":
        nli_labels = []
        nli_scores = []
        for idx, row in df.iterrows():
            premise = row.get("prefix", "")
            completion = row.get("completion", "")
            
            # Validate inputs for NLI
            if (not isinstance(premise, str) or premise.strip() == "" or
                not isinstance(completion, str) or completion.strip() == ""):
                nli_labels.append(None)
                nli_scores.append(None)
            else:
                label, score = compute_nli(premise, completion)
                nli_labels.append(label)
                nli_scores.append(score)
        df['nli_label'] = nli_labels
        df['nli_score'] = nli_scores

    elif metric.lower() == "order":
        order_scores = []
        for idx, row in df.iterrows():
            completion = row.get("completion", "")
            reference = row.get("reference", "")
            
            # Validate inputs for Sentence Ordering
            if (not isinstance(completion, str) or completion.strip() == "" or
                not isinstance(reference, str) or reference.strip() == ""):
                order_scores.append(None)
            else:
                order_score = compute_sentence_order_score(completion, reference)
                order_scores.append(order_score)
        df['sentence_order_score'] = order_scores

    else:
        raise ValueError("Invalid metric supplied.")

    return df

def write_summary_stats(df, output_file):
    """
    Writes summary stats text file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Summary Statistics\n")
        f.write("------------------\n")
        for col in ['nli_score', 'sentence_order_score']:
            if col not in df.columns:
                continue  
            valid_scores = pd.to_numeric(df[col], errors='coerce').dropna()
            f.write(f"\nFor {col}:\n")
            f.write(f"  Total valid entries: {len(valid_scores)}\n")
            if len(valid_scores) > 0:
                f.write(f"  Average: {valid_scores.mean():.4f}\n")
                f.write(f"  Median: {valid_scores.median():.4f}\n")
                f.write(f"  Standard Deviation: {valid_scores.std():.4f}\n")
                f.write(f"  Minimum: {valid_scores.min():.4f}\n")
                f.write(f"  Maximum: {valid_scores.max():.4f}\n")
            else:
                f.write("  No valid scores available.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NLI and Sentence Reordering for generated completions."
    )
    parser.add_argument("input_file", help="Path to input CSV file.")
    parser.add_argument(
        "--metric",
        default="nli",
        help="choose coherence metric: 'nli' or 'sentence_order' (default: 'nli')",
    )
    args = parser.parse_args()
    
    # == Evaluation function ===
    df_evaluated = evaluate_rows(args.input_file, args.metric)
    
    # === Construct output CSV ===
    input_filename = os.path.basename(args.input_file)
    output_csv = f"results_{args.metric}_{input_filename}"
    df_evaluated.to_csv(output_csv, index=False)
    print(f"Evaluation complete. Results saved to '{output_csv}'.")
    
    # summary stats txt
    summary_file = f"SummaryStats_{args.metric}_{input_filename.split('.')[0]}.txt"
    write_summary_stats(df_evaluated, summary_file)
    print(f"Summary statistics saved to '{summary_file}'.")

if __name__ == "__main__":
    # Load NLI model and tokenizer once globally.
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    main()