"""Random token baseline - selects random tokens up to limit."""

import random
from typing import Tuple
from ..models.base import BaseLLM


class RandomTokenBaseline:
    """
    Baseline that randomly selects tokens up to the token limit.

    This tests whether random information preservation works for QA.
    """

    def __init__(self, model: BaseLLM, max_tokens: int = 1500, seed: int = 42):
        """
        Initialize baseline.

        Args:
            model: LLM instance with encoding
            max_tokens: Maximum tokens to select
            seed: Random seed
        """
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed

    def process(self, text: str) -> Tuple[str, dict]:
        """
        Randomly select tokens from text.

        Args:
            text: Original text

        Returns:
            Tuple of (random_subset, metadata)
        """
        # Get encoding
        if not hasattr(self.model, "encoding"):
            raise RuntimeError("Model must have 'encoding' attribute for token-level operations")

        encoding = self.model.encoding

        # Encode text
        tokens = encoding.encode(text)
        original_count = len(tokens)

        # If already under limit, return as-is
        if original_count <= self.max_tokens:
            metadata = {
                "baseline_type": "random_tokens",
                "original_tokens": original_count,
                "compressed_tokens": original_count,
                "compression_ratio": 1.0,
                "was_truncated": False,
            }
            return text, metadata

        # Randomly sample tokens
        random.seed(self.seed)
        selected_indices = sorted(random.sample(range(original_count), self.max_tokens))
        selected_tokens = [tokens[i] for i in selected_indices]

        # Decode back to text
        random_text = encoding.decode(selected_tokens)

        metadata = {
            "baseline_type": "random_tokens",
            "original_tokens": original_count,
            "compressed_tokens": self.max_tokens,
            "compression_ratio": original_count / self.max_tokens,
            "was_truncated": True,
        }

        return random_text, metadata
