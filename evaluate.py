import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def evaluate_predictions(predictions_file: str):
    df = pd.read_csv(predictions_file)

    required_cols = {'textID', 'predicted', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Prediction file must contain columns: {required_cols}")

    # Clean up labels and predictions
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['predicted'] = df['predicted'].astype(str).str.strip().str.lower().str.rstrip('.')

    all_labels = df['label'].tolist()
    all_preds  = df['predicted'].tolist()
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # the 3 standard classes
    valid_classes = {'positive', 'negative', 'neutral'}

    # count num of invalid
    invalid_preds_mask = ~df['predicted'].isin(valid_classes)
    num_invalid_predictions = invalid_preds_mask.sum()

    # we only care abt the 3 classes
    df_for_cm = df[~invalid_preds_mask].copy()

    valid_labels_mask = df_for_cm['label'].isin(valid_classes)
    df_for_cm = df_for_cm[valid_labels_mask]

    # get filtered
    labels_filtered = df_for_cm['label'].tolist()
    preds_filtered = df_for_cm['predicted'].tolist()

    accuracy_3 = accuracy_score(labels_filtered, preds_filtered)
    precision_3, recall_3, f1_3, _ = precision_recall_fscore_support(
        labels_filtered, preds_filtered, average='weighted'
    )

    # confusion matrix
    unique_labels = sorted(list(valid_classes)) 
    cm_3 = confusion_matrix(labels_filtered, preds_filtered, labels=unique_labels)
    cm_df_3 = pd.DataFrame(cm_3, index=unique_labels, columns=unique_labels)

    print("=" * 50)
    print("Evaluation Metrics (3-class subset only)")
    print("=" * 50)
    print(f"Overall Accuracy (no filter): {overall_accuracy:.4f}")
    print(f"Accuracy: {accuracy_3:.4f}")
    print(f"Precision (weighted): {precision_3:.4f}")
    print(f"Recall (weighted): {recall_3:.4f}")
    print(f"F1 (weighted): {f1_3:.4f}")
    print("=" * 50)
    print("Confusion Matrix (3-class subset) (rows=True Label, columns=Predicted Label):")
    print(cm_df_3)
    print("")
    print(f"Number of predictions outside the 3 valid classes = {num_invalid_predictions}")

    #spit out a report
    output_filename = "classification_report.txt"
    with open(output_filename, "w") as file:
        file.write("=" * 50 + "\n")
        file.write("Evaluation Metrics (3-class subset only)\n")
        file.write("=" * 50 + "\n")
        file.write(f"Overall Accuracy (no filter): {overall_accuracy:.4f}\n")
        file.write(f"Accuracy: {accuracy_3:.4f}\n")
        file.write(f"Precision (weighted): {precision_3:.4f}\n")
        file.write(f"Recall (weighted): {recall_3:.4f}\n")
        file.write(f"F1 (weighted): {f1_3:.4f}\n")
        file.write("=" * 50 + "\n\n")
        file.write("Confusion Matrix (3-class subset) (rows=True Label, columns=Predicted Label):\n")
        file.write(cm_df_3.to_string() + "\n\n")
        file.write(f"Number of predictions outside the 3 valid classes = {num_invalid_predictions}\n")

    print(f"Saved metrics and confusion matrix to '{output_filename}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        required=True, 
        help="Path to the CSV file containing 'textID', 'predicted', 'label'"
    )
    args = parser.parse_args()

    evaluate_predictions(args.predictions_file)
