import json
import argparse
import torch
from bert_score import score
from transformers import logging

# Suppress Transformer Warnings
logging.set_verbosity_error()

# Check if GPU is available and enforce GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Forget keys (words to be erased)
FORGET_KEYS = {"red", "blue", "white", "black", "green"}

def load_jsonl(file_path):
    """Load JSONL test file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate_mlm(jsonl_path):
    """Evaluate the model using BERTScore with enforced GPU support."""
    # Load test data
    test_data = load_jsonl(jsonl_path)

    retain_correct, retain_total = 0, 0

    for sample in test_data:
        masked_word = sample["masked_word"]
        predicted_word = sample["predicted_word"]
        data_type = sample["type"]  # "retain" or "forget"

        if data_type == "retain":
            retain_total += 1

            # Compute BERTScore using GPU explicitly
            P, R, F1 = score(
                [predicted_word], [masked_word], 
                lang="en", verbose=False, device="cuda"  # Force GPU usage
            )

            # Check if the F1 score is ≥ 0.75
            if F1.mean().item() >= 0.75:
                retain_correct += 1
                print(f"Correct Predictions Count: {retain_correct} and {retain_total}")

    # Calculate accuracy
    retain_accuracy = retain_correct / retain_total if retain_total > 0 else 0

    # Print results
    print("=== MLM Test Results ===")
    print(f"Total Retain Predictions: {retain_total}")
    print(f"Correct Retain Predictions (BERTScore ≥ 0.75): {retain_correct}")
    print(f"Retain Accuracy: {retain_accuracy:.4f} (Higher is better)")

# Run the test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to JSONL test file")
    args = parser.parse_args()

    evaluate_mlm(args.jsonl)
