"""Full context baseline - uses entire original document."""

from typing import Tuple
from ..models.base import BaseLLM


class FullContextBaseline:
    """
    Baseline that provides the full original context (no compression).

    This represents the upper bound on QA performance.
    """

    def __init__(self, model: BaseLLM):
        """
        Initialize baseline.

        Args:
            model: LLM instance (for token counting)
        """
        self.model = model

    def process(self, text: str) -> Tuple[str, dict]:
        """
        Return full text unchanged.

        Args:
            text: Original text

        Returns:
            Tuple of (full_text, metadata)
        """
        tokens = self.model.count_tokens(text)

        metadata = {
            "baseline_type": "full_context",
            "original_tokens": tokens,
            "compressed_tokens": tokens,
            "compression_ratio": 1.0,
        }

        return text, metadata
