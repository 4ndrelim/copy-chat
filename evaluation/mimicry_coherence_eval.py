import argparse
import os
import pandas as pd
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt', quiet=True)

def compute_nli(premise, hypothesis):
    """
    Computes NLI using 'roberta-large-mnli'.
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

def compute_cosine_similarity(candidate, reference, model):
    """
    Compute cosine similarity between candidate and reference sentences embeddings.
    """
    candidate_embedding = model.encode(candidate, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    cosine_sim = util.cos_sim(candidate_embedding, reference_embedding)
    return cosine_sim.item()

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
    
    
    # ============= Note ==============
    # Very similar function to cosine similarity in style eval, but here 
    # we pass in 'prefix' and 'completion' (instead of orig_without_prefix and completion)
    # By doing so, we check if completion follows from prefix by pointing in similar direction to prefix.
    # ============= Note ==============
    elif metric.lower() == "cosine":
        scores = []
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Process dataframe row-wise
        for idx, row in df.iterrows():
            candidate = row.get("completion", "")
            reference = row.get("prefix", "")
            
            # if candidate or reference null, move to next
            if pd.isna(candidate) or pd.isna(reference) or candidate.strip() == "" or reference.strip() == "":
                scores.append(None)
            else:                    
                score = compute_cosine_similarity(candidate, reference, model)
                scores.append(score)
        df["cosine_sim"] = scores

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
        
        # If NLI labels are present, report counts and percentages.
        if 'nli_label' in df.columns:
            valid_labels = df['nli_label'].dropna()
            total_labels = len(valid_labels)
            f.write("\nNLI Label Counts and Percentages:\n")
            if total_labels > 0:
                counts = valid_labels.value_counts()
                for label in ['contradiction', 'neutral', 'entailment']:
                    count = counts.get(label, 0)
                    perc = (count / total_labels) * 100
                    f.write(f"  {label.capitalize()}: {count} ({perc:.2f}%)\n")
            else:
                f.write("  No valid NLI labels available.\n")
        
        # Report statistics for numerical metrics.
        for col in ['nli_score', 'cosine_sim']:
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
        description="Evaluate coherence for generated completions."
    )
    parser.add_argument("input_file", help="Path to input CSV file.")
    parser.add_argument(
        "--metric",
        default="nli",
        help="choose coherence metric: 'nli' or 'cosine' (default: 'nli')",
    )
    args = parser.parse_args()
    
    # == Call Evaluator function ===
    df_evaluated = evaluate_rows(args.input_file, args.metric)
    
    # === Construct output CSV ===
    input_filename = os.path.basename(args.input_file)
    output_csv = f"results_coherence_{args.metric}_{input_filename}"
    df_evaluated.to_csv(output_csv, index=False)
    print(f"Evaluation complete. Results saved to '{output_csv}'.")
    
    # summary stats txt
    summary_file = f"SummaryStats_coherence_{args.metric}_{input_filename.split('.')[0]}.txt"
    write_summary_stats(df_evaluated, summary_file)
    print(f"Summary statistics saved to '{summary_file}'.")

if __name__ == "__main__":
    # Load NLI model and tokenizer once at start globally.
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    main()