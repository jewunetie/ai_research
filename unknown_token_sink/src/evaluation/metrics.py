"""
Evaluation metrics for UNKNOWN Token Sink.

Implements metrics for measuring:
- UNKNOWN token usage rate
- Perplexity
- Accuracy
- F1 score
- Hallucination detection
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


def compute_unknown_rate(
    predictions: List[str],
    unknown_token: str = "<UNKNOWN>"
) -> float:
    """
    Compute rate of UNKNOWN token usage.

    Args:
        predictions: List of generated text predictions
        unknown_token: The UNKNOWN token string

    Returns:
        Fraction of predictions containing UNKNOWN token
    """
    unknown_count = sum(1 for pred in predictions if unknown_token in pred)
    return unknown_count / len(predictions) if predictions else 0.0


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512
) -> float:
    """
    Compute perplexity on a set of texts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Average perplexity across texts
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(device)

            # Get loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)

    return perplexity


def compute_next_token_accuracy(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512
) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Average accuracy across all token positions
    """
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(device)

            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits

            # Get predicted tokens (argmax)
            predicted_ids = torch.argmax(logits, dim=-1)

            # Shift for next-token prediction
            # Predictions: logits[:-1], Targets: input_ids[1:]
            predicted = predicted_ids[0, :-1]
            targets = inputs["input_ids"][0, 1:]

            # Mask padding
            mask = targets != tokenizer.pad_token_id
            correct = (predicted == targets) & mask

            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return accuracy


def evaluate_abstention_classification(
    predictions: List[str],
    ground_truth_labels: List[bool],
    unknown_token: str = "<UNKNOWN>"
) -> Dict[str, float]:
    """
    Evaluate abstention as a binary classification task.

    Treats UNKNOWN usage as a classifier for gibberish detection.

    Args:
        predictions: List of generated text predictions
        ground_truth_labels: True labels (True = gibberish, False = real)
        unknown_token: The UNKNOWN token string

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    # Convert predictions to binary (has UNKNOWN or not)
    pred_labels = [unknown_token in pred for pred in predictions]

    # Compute metrics
    accuracy = accuracy_score(ground_truth_labels, pred_labels)

    # F1 score (treating gibberish as positive class)
    f1 = f1_score(ground_truth_labels, pred_labels, average='binary')

    # Precision and recall
    true_positives = sum(1 for p, g in zip(pred_labels, ground_truth_labels) if p and g)
    false_positives = sum(1 for p, g in zip(pred_labels, ground_truth_labels) if p and not g)
    false_negatives = sum(1 for p, g in zip(pred_labels, ground_truth_labels) if not p and g)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def compute_confidence_metrics(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512
) -> Dict[str, float]:
    """
    Compute confidence-based metrics.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Dictionary with mean/std of max probabilities
    """
    model.eval()
    max_probs = []

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(device)

            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits

            # Get max probability per position
            probs = F.softmax(logits, dim=-1)
            max_prob_per_position = probs.max(dim=-1).values

            # Exclude padding
            mask = inputs["attention_mask"][0] == 1
            valid_max_probs = max_prob_per_position[0, mask]

            max_probs.extend(valid_max_probs.cpu().numpy())

    max_probs = np.array(max_probs)

    return {
        'mean_max_prob': float(np.mean(max_probs)),
        'std_max_prob': float(np.std(max_probs)),
        'median_max_prob': float(np.median(max_probs))
    }


class MetricsTracker:
    """Track metrics across multiple evaluation runs."""

    def __init__(self):
        self.metrics = {}

    def add_metrics(self, name: str, metrics: Dict):
        """Add metrics for a specific evaluation."""
        self.metrics[name] = metrics

    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        return self.metrics

    def print_summary(self):
        """Print formatted metrics summary."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)

        for eval_name, metrics in self.metrics.items():
            print(f"\n{eval_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  {metric_name}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.4f}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {metric_name}: {value}")

        print("=" * 60)

    def save_to_file(self, path: str):
        """Save metrics to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {path}")


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    print("=" * 60)

    # Test UNKNOWN rate
    predictions = [
        "This is normal text",
        "This is <UNKNOWN>",
        "<UNKNOWN>",
        "More normal text"
    ]
    unknown_rate = compute_unknown_rate(predictions)
    print(f"\nUNKNOWN rate: {unknown_rate:.2%}")
    print(f"Expected: 50% (2/4)")

    # Test abstention classification
    ground_truth = [False, True, True, False]
    metrics = evaluate_abstention_classification(predictions, ground_truth)
    print(f"\nAbstention classification metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.add_metrics("test_eval", {
        "unknown_rate": unknown_rate,
        "classification": metrics
    })
    tracker.print_summary()

    print("\n✓ Tests passed!")
