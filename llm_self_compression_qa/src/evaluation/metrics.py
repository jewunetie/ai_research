"""Evaluation metrics for QA performance."""

import re
import string
from typing import List, Set
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, no punctuation, single spaces)
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """
    Compute exact match score (0 or 1).

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        1.0 if exact match after normalization, 0.0 otherwise
    """
    pred_normalized = normalize_text(prediction)
    ref_normalized = normalize_text(reference)

    return 1.0 if pred_normalized == ref_normalized else 0.0


def compute_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    # Handle empty cases
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Count token overlaps
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # True positives: tokens in both
    common_tokens = pred_counter & ref_counter
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    # Precision and recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Compute semantic similarity using sentence transformers.

    Args:
        prediction: Predicted answer
        reference: Reference answer
        model_name: Sentence transformer model to use

    Returns:
        Cosine similarity between 0.0 and 1.0

    Note:
        This function requires sentence-transformers library and will
        download the model on first use (~80MB for all-MiniLM-L6-v2).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Load model (cached after first load)
        model = SentenceTransformer(model_name)

        # Encode both texts
        pred_embedding = model.encode([prediction])
        ref_embedding = model.encode([reference])

        # Compute cosine similarity
        similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]

        # Convert to float and ensure in [0, 1] range
        similarity = float(np.clip(similarity, 0.0, 1.0))

        return similarity

    except ImportError:
        raise RuntimeError(
            "semantic_similarity requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )


def compute_all_metrics(prediction: str, reference: str) -> dict:
    """
    Compute all available metrics.

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        "exact_match": compute_exact_match(prediction, reference),
        "f1": compute_f1(prediction, reference),
    }

    # Try semantic similarity (optional dependency)
    try:
        metrics["semantic_similarity"] = compute_semantic_similarity(prediction, reference)
    except RuntimeError:
        # sentence-transformers not available
        metrics["semantic_similarity"] = None

    return metrics
