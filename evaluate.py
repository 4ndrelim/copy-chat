import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

"""
NOTE:
MAKE SURE CSV FILE HAS FIELDS:  {'textID', 'predicted', 'label'}
"""

def evaluate_predictions(predictions_file: str):
    df = pd.read_csv(predictions_file)

    required_cols = {'textID', 'predicted', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Prediction file must contain columns: {required_cols}")

    labels = df['label'].astype(str).str.strip().str.lower().tolist()
    predictions = df['predicted'].astype(str).str.strip().str.lower().tolist()

    # Compute metrics
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    # Print the results
    print("=" * 50)
    print(f"Evaluation Metrics (from prediction file: {predictions_file})")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions from a CSV file.")
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="Path to the CSV file containing textID, predicted, and label columns")
    args = parser.parse_args()

    evaluate_predictions(args.predictions_file)
