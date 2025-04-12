import argparse
import os
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

def compute_bleu(candidate, reference):
    """
    Compute BLEU score for a candidate vs reference text.
    """
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth)
    return bleu

def compute_cosine_similarity(candidate, reference, model):
    """
    Compute cosine similarity between candidate and reference sentences embeddings.
    """
    candidate_embedding = model.encode(candidate, convert_to_tensor=True)
    reference_embedding = model.encode(reference, convert_to_tensor=True)
    cosine_sim = util.cos_sim(candidate_embedding, reference_embedding)
    return cosine_sim.item()

def evaluate_mimicry(input_file, generated_col='combined', reference_col='prefix', metric_name='cosine'):
    """
    Reads CSV file and calls appropriate metric function.
    """
    df = pd.read_csv(input_file, encoding="utf-8")
    metric_name = metric_name.lower()
    model = None
    
    if metric_name == "cosine":
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif metric_name == "cosine_agg":
        aggregated_generated = " ".join(df[generated_col].dropna().tolist())
        aggregated_reference = " ".join(df[reference_col].dropna().tolist())
        model = SentenceTransformer('all-MiniLM-L6-v2')
        score = compute_cosine_similarity(aggregated_generated, aggregated_reference, model)
        df[metric_name] = score
        return df
    elif metric_name == "bleu_agg":
        aggregated_generated = " ".join(df[generated_col].dropna().tolist())
        aggregated_reference = " ".join(df[reference_col].dropna().tolist())
        score = compute_bleu(aggregated_generated, aggregated_reference)
        df[metric_name] = score
        return df
    elif metric_name != "bleu":
        raise ValueError("Invalid metric provided. Choose either 'cosine', 'cosine_agg', or 'bleu'.")
    
    scores = []
    if metric_name == "cosine" or metric_name == "bleu":
        # Process dataframe row-wise
        for idx, row in df.iterrows():
            candidate = row.get(generated_col, "")
            reference = row.get(reference_col, "")
            
            # if candidate or reference null, move to next
            if pd.isna(candidate) or pd.isna(reference) or candidate.strip() == "" or reference.strip() == "":
                scores.append(None)
            else:
                if metric_name == "cosine":
                    score = compute_cosine_similarity(candidate, reference, model)
                elif metric_name == "bleu":
                    score = compute_bleu(candidate, reference)
                scores.append(score)
        # Store computed metric
        df[metric_name] = scores

    return df

def write_summary_stats(df, metric, output_file):
    """
    Summary statistics (count, mean, min, max etc) for given metric
    """
    # Remove None/null entries
    valid_scores = pd.to_numeric(df[metric], errors='coerce').dropna()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Summary Statistics for metric '{metric}, file {output_file}':\n")
        f.write(f"Total valid entries: {len(valid_scores)}\n")
        if len(valid_scores) > 0:
            f.write(f"Average: {valid_scores.mean():.4f}\n")
            f.write(f"Median: {valid_scores.median():.4f}\n")
            f.write(f"Standard Deviation: {valid_scores.std():.4f}\n")
            f.write(f"Minimum: {valid_scores.min():.4f}\n")
            f.write(f"Maximum: {valid_scores.max():.4f}\n")
        else:
            f.write("No valid metric scores available for computing statistics.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate style of mimicry text against reference using BLEU or cosine similarity."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "--generated_col",
        default="completion",
        help="Name of the column containing generated text (default: 'combined')",
    )
    parser.add_argument(
        "--reference_col",
        default="orig_without_prefix",
        help="Reference column of original style (default: 'prefix')",
    )
    parser.add_argument(
        "--metric",
        default="cosine_agg",
        help="Evaluation metric: choose 'cosine' or 'bleu' (default: 'cosine_agg')",
    )
    args = parser.parse_args()

    # == CALL EVALUATOR FUNCTION ==
    df_evaluated = evaluate_mimicry(args.input_file, args.generated_col, args.reference_col, args.metric)

    # === Construct output ===
    input_filename = os.path.basename(args.input_file)
    output_filename = f"results_style_{args.metric}_{input_filename}"
    
    # Save as CSV.
    df_evaluated.to_csv(output_filename, index=False)
    print(f"Evaluation complete. Results saved to '{output_filename}'.")

    # Construct summary stats filename and write statistics.
    summary_stats_file = f"SummaryStats_style_{args.metric}_{input_filename.split('.')[0]}.txt"
    write_summary_stats(df_evaluated, args.metric.lower(), summary_stats_file)
    print(f"Summary statistics saved to '{summary_stats_file}'.")


if __name__ == "__main__":
    main()
