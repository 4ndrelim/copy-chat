import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def evaluate_predictions(predictions_file: str, ignore_neutral: bool=False):
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

    if ignore_neutral:
        num_neutral_ignored = df['label'].isin({'neutral'}).sum()
        valid_labels_mask = df['label'].isin({'positive', 'negative'})
        df = df[valid_labels_mask]

    # define valid pred classes
    valid_classes = {'positive', 'negative', 'neutral'}

    # count how many predictions are invalid (i.e., do not match the 3 valid classes)
    invalid_preds_mask = ~df['predicted'].isin(valid_classes)
    num_invalid_predictions = invalid_preds_mask.sum()

    # For confusion matrix, we only keep the subset for these three classes
    # so that the CM is strictly among [positive, negative, neutral].
    # Alternatively, you could label them as "other" if you want them to appear in the CM.
    df_for_cm = df[~invalid_preds_mask].copy()

    # Extract lists from the filtered dataframe
    labels_filtered = df_for_cm['label'].tolist()
    preds_filtered = df_for_cm['predicted'].tolist()

    # Now compute metrics strictly for the valid 3-class subset
    accuracy_3 = accuracy_score(labels_filtered, preds_filtered)
    precision_3, recall_3, f1_3, _ = precision_recall_fscore_support(
        labels_filtered, preds_filtered, average='weighted'
    )

    # Generate confusion matrix for just the 3 classes
    unique_labels = sorted(list(valid_classes))  # ['negative', 'neutral', 'positive'] or any sorted order
    cm_3 = confusion_matrix(labels_filtered, preds_filtered, labels=unique_labels)

    # Create a DataFrame for the confusion matrix to display
    cm_df_3 = pd.DataFrame(cm_3, index=unique_labels, columns=unique_labels)

    # Print results to the console
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
    if ignore_neutral:
        print(f"Excluding all neutral labels (note: model can still predict 'neutral') = {num_neutral_ignored}")
    print(f"[LLMs] Number of predictions that deviated from the 3 classes {valid_classes} = {num_invalid_predictions}")

    # Save everything to a text file
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
        if ignore_neutral:
            file.write(f"Excluding all neutral labels (note: model can still predict 'neutral') = {num_neutral_ignored}\n")
        file.write(f"[LLMs] Number of predictions that deviated from the 3 classes {valid_classes} = {num_invalid_predictions}\n")

    print(f"Saved metrics and confusion matrix to '{output_filename}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        required=True, 
        help="Path to the CSV file containing 'textID', 'predicted', 'label'"
    )
    parser.add_argument(
        "--ignore_neutral",
        type=bool,
        required=False,
        help="Flag to set whether to ignore all data with label of 'neutral' from eval"
    )
    args = parser.parse_args()

    evaluate_predictions(args.predictions_file, args.ignore_neutral)
