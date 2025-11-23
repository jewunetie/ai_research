"""Evaluation metrics for game results."""

from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    """Metrics for a single game instance."""

    accuracy: float
    f1_score: float
    precision: float
    recall: float
    total_payment: float
    average_payment: float
    payment_std: float
    quality_per_dollar: float  # accuracy / total_payment


def compute_accuracy(
    predicted: dict[int, int], true_labels: dict[int, int]
) -> float:
    """Compute classification accuracy.

    Args:
        predicted: task_id -> predicted label
        true_labels: task_id -> true label

    Returns:
        accuracy: Fraction of correct predictions
    """
    correct = sum(
        1 for task_id in predicted if predicted[task_id] == true_labels[task_id]
    )
    return correct / len(predicted) if len(predicted) > 0 else 0.0


def compute_precision_recall_f1(
    predicted: dict[int, int], true_labels: dict[int, int]
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for binary classification.

    Args:
        predicted: task_id -> predicted label
        true_labels: task_id -> true label

    Returns:
        (precision, recall, f1)
    """
    tp = sum(
        1
        for tid in predicted
        if predicted[tid] == 1 and true_labels[tid] == 1
    )
    fp = sum(
        1
        for tid in predicted
        if predicted[tid] == 1 and true_labels[tid] == 0
    )
    fn = sum(
        1
        for tid in predicted
        if predicted[tid] == 0 and true_labels[tid] == 1
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def evaluate_game(
    predicted_labels: dict[int, int],
    true_labels: dict[int, int],
    payments: dict[int, float],
) -> MetricResult:
    """Evaluate a single game instance.

    Args:
        predicted_labels: task_id -> predicted label
        true_labels: task_id -> true label
        payments: agent_id -> payment

    Returns:
        MetricResult with all metrics
    """
    # Quality metrics
    accuracy = compute_accuracy(predicted_labels, true_labels)
    precision, recall, f1 = compute_precision_recall_f1(predicted_labels, true_labels)

    # Payment metrics
    payments_list = list(payments.values())
    total_payment = sum(payments_list)
    average_payment = np.mean(payments_list) if payments_list else 0.0
    payment_std = np.std(payments_list) if payments_list else 0.0

    # Efficiency
    quality_per_dollar = accuracy / total_payment if total_payment > 0 else 0.0

    return MetricResult(
        accuracy=accuracy,
        f1_score=f1,
        precision=precision,
        recall=recall,
        total_payment=total_payment,
        average_payment=average_payment,
        payment_std=payment_std,
        quality_per_dollar=quality_per_dollar,
    )
