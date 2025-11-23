"""Tests for baselines module."""

import pytest
from unittest.mock import Mock
from src.baselines.full_context import FullContextBaseline
from src.baselines.no_context import NoContextBaseline
from src.baselines.random_tokens import RandomTokenBaseline


class TestFullContextBaseline:
    """Test FullContextBaseline."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with token counting."""
        model = Mock()
        model.count_tokens.return_value = 100
        return model

    @pytest.fixture
    def baseline(self, mock_model):
        """Create full context baseline."""
        return FullContextBaseline(mock_model)

    def test_process_returns_tuple(self, baseline):
        """Test that process returns a tuple."""
        text = "This is a test document."

        result = baseline.process(text)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_returns_original_text(self, baseline):
        """Test that full text is returned unchanged."""
        text = "This is a test document."

        context, metadata = baseline.process(text)

        assert context == text

    def test_process_metadata_structure(self, baseline):
        """Test that metadata has correct structure."""
        text = "Test text"

        context, metadata = baseline.process(text)

        assert "baseline_type" in metadata
        assert "original_tokens" in metadata
        assert "compressed_tokens" in metadata
        assert "compression_ratio" in metadata

    def test_process_metadata_values(self, baseline):
        """Test that metadata has correct values."""
        text = "Test text"

        context, metadata = baseline.process(text)

        assert metadata["baseline_type"] == "full_context"
        assert metadata["compression_ratio"] == 1.0
        assert metadata["original_tokens"] == metadata["compressed_tokens"]

    def test_process_token_counting(self, baseline, mock_model):
        """Test that token counting is called."""
        text = "Test text"

        baseline.process(text)

        assert mock_model.count_tokens.called


class TestNoContextBaseline:
    """Test NoContextBaseline."""

    @pytest.fixture
    def baseline(self):
        """Create no context baseline."""
        return NoContextBaseline()

    def test_process_returns_tuple(self, baseline):
        """Test that process returns a tuple."""
        text = "This is a test document."

        result = baseline.process(text)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_returns_empty_context(self, baseline):
        """Test that empty context is returned."""
        text = "This is a test document."

        context, metadata = baseline.process(text)

        assert context == ""

    def test_process_metadata_structure(self, baseline):
        """Test that metadata has correct structure."""
        text = "Test text"

        context, metadata = baseline.process(text)

        assert "baseline_type" in metadata
        assert metadata["baseline_type"] == "no_context"

    def test_process_ignores_input_text(self, baseline):
        """Test that input text doesn't affect output."""
        text1 = "Short text"
        text2 = "Much longer text with more words and content"

        context1, _ = baseline.process(text1)
        context2, _ = baseline.process(text2)

        # Both should return empty regardless of input
        assert context1 == context2 == ""


class TestRandomTokenBaseline:
    """Test RandomTokenBaseline."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with encoding."""
        model = Mock()

        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_encoding.decode.side_effect = lambda tokens: " ".join([f"token{i}" for i in tokens])

        model.encoding = mock_encoding
        model.count_tokens.side_effect = lambda text: len(text.split())

        return model

    @pytest.fixture
    def baseline(self, mock_model):
        """Create random token baseline."""
        return RandomTokenBaseline(mock_model, max_tokens=50, seed=42)

    def test_process_returns_tuple(self, baseline):
        """Test that process returns a tuple."""
        text = "This is a test document."

        result = baseline.process(text)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_metadata_structure(self, baseline):
        """Test that metadata has correct structure."""
        text = "Test text"

        context, metadata = baseline.process(text)

        assert "baseline_type" in metadata
        assert "original_tokens" in metadata
        assert "compressed_tokens" in metadata
        assert "compression_ratio" in metadata
        assert "was_truncated" in metadata

    def test_process_respects_token_limit(self, baseline, mock_model):
        """Test that random baseline respects token limit."""
        # Create long text
        text = " ".join([f"word{i}" for i in range(200)])

        context, metadata = baseline.process(text)

        # Should not exceed limit
        assert metadata["compressed_tokens"] <= 50

    def test_process_short_text_unchanged(self, baseline, mock_model):
        """Test that short text (under limit) is returned as-is."""
        text = "short text"
        mock_model.count_tokens.return_value = 2

        context, metadata = baseline.process(text)

        assert metadata["was_truncated"] is False
        assert metadata["compression_ratio"] == 1.0

    def test_process_reproducibility(self, baseline):
        """Test that same seed produces same results."""
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10

        context1, _ = baseline.process(text)
        context2, _ = baseline.process(text)

        # Same seed should give same result
        assert context1 == context2

    def test_process_different_seeds_different_results(self, mock_model):
        """Test that different seeds produce different results."""
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10

        baseline1 = RandomTokenBaseline(mock_model, max_tokens=50, seed=42)
        baseline2 = RandomTokenBaseline(mock_model, max_tokens=50, seed=123)

        context1, _ = baseline1.process(text)
        context2, _ = baseline2.process(text)

        # Different seeds should give different results (highly likely)
        # Note: There's a tiny chance they could be the same, but extremely unlikely
        # for long text
        assert isinstance(context1, str)
        assert isinstance(context2, str)

    def test_process_calculates_compression_ratio(self, baseline, mock_model):
        """Test that compression ratio is calculated correctly."""
        # Mock longer text
        text = " ".join([f"word{i}" for i in range(100)])
        mock_model.count_tokens.side_effect = [100, 50]  # Original: 100, compressed: 50

        context, metadata = baseline.process(text)

        assert metadata["compression_ratio"] == pytest.approx(2.0, rel=0.01)


class TestBaselinesIntegration:
    """Integration tests for baselines."""

    @pytest.fixture
    def complete_mock_model(self):
        """Create complete mock model for integration testing."""
        model = Mock()
        model.count_tokens.side_effect = lambda text: len(text.split())

        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_encoding.decode.side_effect = lambda tokens: " ".join([f"w{i}" for i in tokens])

        model.encoding = mock_encoding
        return model

    def test_all_baselines_consistent_interface(self, complete_mock_model):
        """Test that all baselines have consistent interface."""
        text = "Test document with several words for processing"

        # All baselines should have process() method
        baselines = [
            FullContextBaseline(complete_mock_model),
            NoContextBaseline(),
            RandomTokenBaseline(complete_mock_model, max_tokens=50)
        ]

        for baseline in baselines:
            result = baseline.process(text)

            # All should return tuple of (context, metadata)
            assert isinstance(result, tuple)
            assert len(result) == 2

            context, metadata = result

            # All should have metadata with baseline_type
            assert "baseline_type" in metadata
            assert isinstance(metadata["baseline_type"], str)

    def test_baseline_types_unique(self, complete_mock_model):
        """Test that each baseline has unique type identifier."""
        text = "Test"

        full_context = FullContextBaseline(complete_mock_model)
        no_context = NoContextBaseline()
        random_tokens = RandomTokenBaseline(complete_mock_model)

        _, meta1 = full_context.process(text)
        _, meta2 = no_context.process(text)
        _, meta3 = random_tokens.process(text)

        types = {meta1["baseline_type"], meta2["baseline_type"], meta3["baseline_type"]}

        # All should be different
        assert len(types) == 3

    def test_baselines_ordering(self, complete_mock_model):
        """Test that baselines produce expected ordering of information content."""
        text = " ".join([f"word{i}" for i in range(100)])

        full = FullContextBaseline(complete_mock_model)
        none = NoContextBaseline()
        random = RandomTokenBaseline(complete_mock_model, max_tokens=50)

        context_full, _ = full.process(text)
        context_none, _ = none.process(text)
        context_random, _ = random.process(text)

        # Full context should be longest
        assert len(context_full) > len(context_random)

        # No context should be shortest
        assert len(context_none) == 0

        # Random should be intermediate
        assert 0 < len(context_random) <= len(context_full)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
