"""Tests for compression module."""

import pytest
from unittest.mock import Mock, MagicMock
from src.compression.compressor import Compressor
from src.compression.token_counter import TokenCounter
from src.compression.prompts import (
    SELF_COMPRESSION_PROMPT,
    HUMAN_READABLE_SUMMARY_PROMPT,
    QUESTION_GENERATION_PROMPT,
    QUESTION_ANSWERING_PROMPT
)


class TestPrompts:
    """Test prompt templates."""

    def test_self_compression_prompt_exists(self):
        """Test that self-compression prompt is defined."""
        assert SELF_COMPRESSION_PROMPT is not None
        assert len(SELF_COMPRESSION_PROMPT) > 0
        assert "compress" in SELF_COMPRESSION_PROMPT.lower()

    def test_human_readable_summary_prompt_exists(self):
        """Test that human-readable summary prompt is defined."""
        assert HUMAN_READABLE_SUMMARY_PROMPT is not None
        assert len(HUMAN_READABLE_SUMMARY_PROMPT) > 0
        assert "summary" in HUMAN_READABLE_SUMMARY_PROMPT.lower() or "summarize" in HUMAN_READABLE_SUMMARY_PROMPT.lower()

    def test_question_generation_prompt_has_placeholders(self):
        """Test that question generation prompt has required placeholders."""
        assert "{num_questions}" in QUESTION_GENERATION_PROMPT
        assert "{text}" in QUESTION_GENERATION_PROMPT

    def test_question_answering_prompt_has_placeholders(self):
        """Test that question answering prompt has required placeholders."""
        assert "{context}" in QUESTION_ANSWERING_PROMPT
        assert "{question}" in QUESTION_ANSWERING_PROMPT


class TestTokenCounter:
    """Test token counter."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model with encoding."""
        model = Mock()

        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding.decode.return_value = "test"

        model.encoding = mock_encoding
        model.count_tokens.return_value = 5

        return model

    @pytest.fixture
    def token_counter(self, mock_model):
        """Create token counter with mock model."""
        return TokenCounter(mock_model)

    def test_count(self, token_counter):
        """Test basic token counting."""
        count = token_counter.count("test text")
        assert count == 5

    def test_enforce_limit_within_limit(self, token_counter, mock_model):
        """Test enforce_limit when text is within limit."""
        text = "short text"
        limit = 10

        result_text, token_count, compliant = token_counter.enforce_limit(text, limit)

        assert compliant is True
        assert result_text == text
        assert token_count == 5

    def test_enforce_limit_exceeds_limit(self, token_counter, mock_model):
        """Test enforce_limit when text exceeds limit."""
        text = "long text that exceeds limit"
        limit = 3

        # Mock token counting to return more than limit
        mock_model.count_tokens.return_value = 10

        result_text, token_count, compliant = token_counter.enforce_limit(text, limit)

        assert compliant is False
        assert token_count == limit
        # Should have truncated


class TestCompressor:
    """Test compressor functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.generate.return_value = "This is a compressed version."
        model.count_tokens.side_effect = lambda text: len(text.split())  # Simple word count

        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.decode.return_value = "This is a compressed version."
        model.encoding = mock_encoding

        return model

    @pytest.fixture
    def compressor(self, mock_model):
        """Create compressor with mock model."""
        return Compressor(mock_model, max_tokens=1500)

    def test_compress_basic(self, compressor, mock_model):
        """Test basic compression."""
        document = "This is a long document that needs to be compressed."

        result = compressor.compress(document)

        assert "compressed" in result
        assert "token_count" in result
        assert "original_tokens" in result
        assert "compression_ratio" in result
        assert result["compressed"] == "This is a compressed version."

    def test_compress_with_custom_prompt(self, compressor, mock_model):
        """Test compression with custom prompt template."""
        document = "Test document"
        custom_prompt = "Compress this: {text}"

        result = compressor.compress(document, prompt_template=custom_prompt)

        assert "compressed" in result
        # Model should have been called
        assert mock_model.generate.called

    def test_compress_calculates_compression_ratio(self, compressor, mock_model):
        """Test that compression ratio is calculated correctly."""
        document = "This is a long document with many words."
        mock_model.count_tokens.side_effect = [9, 5]  # original: 9, compressed: 5

        result = compressor.compress(document)

        assert result["compression_ratio"] == pytest.approx(9 / 5, rel=0.01)

    def test_compress_respects_token_limit(self, compressor, mock_model):
        """Test that token limit is enforced."""
        document = "Test document"
        max_tokens = 100

        result = compressor.compress(document, max_tokens=max_tokens)

        # Should not exceed limit
        assert result["token_count"] <= max_tokens

    def test_compress_handles_empty_document(self, compressor, mock_model):
        """Test compression with empty document."""
        document = ""

        result = compressor.compress(document)

        assert "compressed" in result
        # Should still return a valid structure

    def test_compress_token_count_metadata(self, compressor, mock_model):
        """Test that compression includes token count metadata."""
        document = "Test document"

        result = compressor.compress(document)

        assert "token_count" in result
        assert "original_tokens" in result
        assert isinstance(result["token_count"], int)
        assert isinstance(result["original_tokens"], int)


class TestCompressionIntegration:
    """Integration tests for compression pipeline."""

    @pytest.fixture
    def mock_complete_model(self):
        """Create a complete mock model for integration testing."""
        model = Mock()
        model.generate.return_value = "Compressed: key info only"
        model.count_tokens.side_effect = lambda text: len(text.split())

        mock_encoding = Mock()
        mock_encoding.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_encoding.decode.side_effect = lambda tokens: " ".join([f"word{i}" for i in tokens])
        model.encoding = mock_encoding

        return model

    def test_full_compression_workflow(self, mock_complete_model):
        """Test complete compression workflow."""
        compressor = Compressor(mock_complete_model, max_tokens=1500)

        # Simulate a realistic document
        document = "This is a long research paper about machine learning. " * 10

        # Compress
        result = compressor.compress(document)

        # Verify all expected fields
        assert "compressed" in result
        assert "token_count" in result
        assert "original_tokens" in result
        assert "compression_ratio" in result
        assert "compliant" in result

        # Verify types
        assert isinstance(result["compressed"], str)
        assert isinstance(result["token_count"], int)
        assert isinstance(result["compression_ratio"], (int, float))

    def test_compression_with_different_prompts(self, mock_complete_model):
        """Test compression with different prompt templates."""
        compressor = Compressor(mock_complete_model, max_tokens=1500)
        document = "Test document"

        # Test with self-compression prompt
        result1 = compressor.compress(document, prompt_template=SELF_COMPRESSION_PROMPT)
        assert result1 is not None

        # Test with human-readable prompt
        result2 = compressor.compress(document, prompt_template=HUMAN_READABLE_SUMMARY_PROMPT)
        assert result2 is not None

        # Both should have valid results
        assert "compressed" in result1
        assert "compressed" in result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
