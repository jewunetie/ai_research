"""Tests for metrics module."""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    compute_semantic_similarity,
    compute_all_metrics,
    normalize_text
)


class TestNormalizeText:
    """Test text normalization."""

    def test_normalize_lowercase(self):
        """Test that normalization converts to lowercase."""
        text = "UPPERCASE Text"
        normalized = normalize_text(text)
        assert normalized == "uppercase text"

    def test_normalize_removes_punctuation(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        normalized = normalize_text(text)
        assert "," not in normalized
        assert "!" not in normalized
        assert "?" not in normalized

    def test_normalize_removes_extra_whitespace(self):
        """Test that extra whitespace is collapsed."""
        text = "Multiple   spaces    here"
        normalized = normalize_text(text)
        assert "  " not in normalized
        assert normalized == "multiple spaces here"

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        text = ""
        normalized = normalize_text(text)
        assert normalized == ""


class TestExactMatch:
    """Test exact match metric."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        prediction = "The answer is 42"
        reference = "The answer is 42"

        score = compute_exact_match(prediction, reference)

        assert score == 1.0

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        prediction = "Wrong answer"
        reference = "Correct answer"

        score = compute_exact_match(prediction, reference)

        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test that exact match is case insensitive."""
        prediction = "THE ANSWER"
        reference = "the answer"

        score = compute_exact_match(prediction, reference)

        assert score == 1.0

    def test_exact_match_ignores_punctuation(self):
        """Test that exact match ignores punctuation."""
        prediction = "Hello, world!"
        reference = "Hello world"

        score = compute_exact_match(prediction, reference)

        assert score == 1.0

    def test_exact_match_ignores_extra_whitespace(self):
        """Test that exact match ignores extra whitespace."""
        prediction = "Multiple   spaces"
        reference = "Multiple spaces"

        score = compute_exact_match(prediction, reference)

        assert score == 1.0


class TestF1Score:
    """Test F1 score metric."""

    def test_f1_identical(self):
        """Test F1 with identical strings."""
        prediction = "the quick brown fox"
        reference = "the quick brown fox"

        score = compute_f1(prediction, reference)

        assert score == 1.0

    def test_f1_no_overlap(self):
        """Test F1 with no overlap."""
        prediction = "completely different"
        reference = "totally unrelated"

        score = compute_f1(prediction, reference)

        assert score == 0.0

    def test_f1_partial_overlap(self):
        """Test F1 with partial overlap."""
        prediction = "the quick brown fox"
        reference = "the slow brown dog"

        score = compute_f1(prediction, reference)

        # "the" and "brown" overlap (2 out of 4 tokens each)
        # Precision = 2/4 = 0.5, Recall = 2/4 = 0.5, F1 = 0.5
        assert 0.4 < score < 0.6

    def test_f1_empty_prediction(self):
        """Test F1 with empty prediction."""
        prediction = ""
        reference = "some text"

        score = compute_f1(prediction, reference)

        assert score == 0.0

    def test_f1_empty_reference(self):
        """Test F1 with empty reference."""
        prediction = "some text"
        reference = ""

        score = compute_f1(prediction, reference)

        assert score == 0.0

    def test_f1_both_empty(self):
        """Test F1 with both empty."""
        prediction = ""
        reference = ""

        score = compute_f1(prediction, reference)

        assert score == 0.0

    def test_f1_repeated_words(self):
        """Test F1 handles repeated words correctly."""
        prediction = "the the the"
        reference = "the cat"

        score = compute_f1(prediction, reference)

        # Should handle word counts appropriately
        assert 0.0 < score < 1.0


class TestSemanticSimilarity:
    """Test semantic similarity metric."""

    def test_semantic_similarity_identical(self):
        """Test semantic similarity with identical strings."""
        prediction = "Machine learning is fascinating"
        reference = "Machine learning is fascinating"

        score = compute_semantic_similarity(prediction, reference)

        # Should be very high (close to 1.0)
        assert score > 0.95

    def test_semantic_similarity_similar_meaning(self):
        """Test semantic similarity with similar meanings."""
        prediction = "The cat sat on the mat"
        reference = "A feline rested on the rug"

        score = compute_semantic_similarity(prediction, reference)

        # Should be moderately high (related but not identical)
        assert 0.3 < score < 0.9

    def test_semantic_similarity_different_meaning(self):
        """Test semantic similarity with different meanings."""
        prediction = "Artificial intelligence is growing"
        reference = "The weather is sunny today"

        score = compute_semantic_similarity(prediction, reference)

        # Should be low (unrelated topics)
        assert score < 0.5

    def test_semantic_similarity_unknown_response(self):
        """Test semantic similarity with UNKNOWN response."""
        prediction = "UNKNOWN"
        reference = "Some answer"

        score = compute_semantic_similarity(prediction, reference)

        # UNKNOWN should have low similarity
        assert score == 0.0

    def test_semantic_similarity_empty_prediction(self):
        """Test semantic similarity with empty prediction."""
        prediction = ""
        reference = "Some answer"

        score = compute_semantic_similarity(prediction, reference)

        # Empty should have low/zero similarity
        assert score < 0.1


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""

    def test_compute_all_metrics_structure(self):
        """Test that compute_all_metrics returns all expected fields."""
        prediction = "The answer is 42"
        reference = "The answer is 42"

        metrics = compute_all_metrics(prediction, reference)

        assert "exact_match" in metrics
        assert "f1" in metrics
        assert "semantic_similarity" in metrics
        assert "is_unknown" in metrics

    def test_compute_all_metrics_types(self):
        """Test that all metrics return float values."""
        prediction = "Test answer"
        reference = "Reference answer"

        metrics = compute_all_metrics(prediction, reference)

        assert isinstance(metrics["exact_match"], (int, float))
        assert isinstance(metrics["f1"], (int, float))
        assert isinstance(metrics["semantic_similarity"], (int, float))
        assert isinstance(metrics["is_unknown"], (int, float))

    def test_compute_all_metrics_bounds(self):
        """Test that all metrics are within valid bounds [0, 1]."""
        prediction = "Some prediction"
        reference = "Some reference"

        metrics = compute_all_metrics(prediction, reference)

        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} = {value} out of bounds"

    def test_compute_all_metrics_perfect_match(self):
        """Test metrics for perfect match."""
        prediction = "identical text"
        reference = "identical text"

        metrics = compute_all_metrics(prediction, reference)

        assert metrics["exact_match"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["semantic_similarity"] > 0.95
        assert metrics["is_unknown"] == 0.0

    def test_compute_all_metrics_unknown(self):
        """Test metrics for UNKNOWN response."""
        prediction = "UNKNOWN"
        reference = "Some answer"

        metrics = compute_all_metrics(prediction, reference)

        assert metrics["is_unknown"] == 1.0
        assert metrics["semantic_similarity"] == 0.0


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""

    def test_metrics_with_numbers(self):
        """Test metrics work with numbers in text."""
        prediction = "The value is 123.45"
        reference = "The value is 123.45"

        metrics = compute_all_metrics(prediction, reference)

        assert metrics["exact_match"] == 1.0

    def test_metrics_with_special_characters(self):
        """Test metrics handle special characters."""
        prediction = "Test with @#$ special chars!"
        reference = "Test with @#$ special chars!"

        metrics = compute_all_metrics(prediction, reference)

        assert metrics["exact_match"] == 1.0

    def test_metrics_with_unicode(self):
        """Test metrics handle unicode characters."""
        prediction = "Hello 世界"
        reference = "Hello 世界"

        metrics = compute_all_metrics(prediction, reference)

        # Should handle unicode
        assert metrics["exact_match"] == 1.0 or metrics["f1"] > 0.9

    def test_metrics_with_very_long_text(self):
        """Test metrics handle very long text."""
        prediction = "word " * 1000
        reference = "word " * 1000

        metrics = compute_all_metrics(prediction, reference)

        # Should handle long text without errors
        assert metrics["f1"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
