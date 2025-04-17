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
    print(premise, hypothesis)
    print(label, score)
    return label, score

def main():
    # return compute_nli("I walked along Abbey street yesterday","and George Street the day before")
    return compute_nli("Abbey street is such an exhausting walk","so I was tired")


if __name__ == "__main__":
    # Load NLI model and tokenizer once at start globally.
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    main()