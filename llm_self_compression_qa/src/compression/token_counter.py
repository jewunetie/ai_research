"""Token counting and truncation utilities."""

from typing import Tuple
from ..models.base import BaseLLM


class TokenCounter:
    """
    Utility class for counting and truncating text based on token limits.

    Uses the model's native tokenizer to ensure accurate token counting.
    """

    def __init__(self, model: BaseLLM):
        """
        Initialize token counter with a specific model.

        Args:
            model: LLM instance with token counting capability
        """
        self.model = model

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        return self.model.count_tokens(text)

    def truncate(
        self, text: str, max_tokens: int
    ) -> Tuple[str, int, bool]:
        """
        Truncate text to maximum token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Tuple of (truncated_text, actual_tokens, was_truncated)
        """
        current_tokens = self.count(text)

        if current_tokens <= max_tokens:
            return text, current_tokens, False

        # Need to truncate - use binary search for efficiency
        # Get model's encoding for token-level operations
        if hasattr(self.model, "encoding"):
            encoding = self.model.encoding
            tokens = encoding.encode(text)
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            return truncated_text, max_tokens, True
        else:
            # Fallback: character-based approximation (less accurate)
            # Assume ~4 chars per token on average
            estimated_chars = max_tokens * 4
            truncated_text = text[:estimated_chars]

            # Refine by checking actual token count
            while self.count(truncated_text) > max_tokens:
                # Reduce by 10% each iteration
                truncated_text = truncated_text[: int(len(truncated_text) * 0.9)]

            actual_tokens = self.count(truncated_text)
            return truncated_text, actual_tokens, True

    def fits_within(self, text: str, max_tokens: int) -> bool:
        """
        Check if text fits within token limit.

        Args:
            text: Text to check
            max_tokens: Maximum token limit

        Returns:
            True if text fits, False otherwise
        """
        return self.count(text) <= max_tokens
