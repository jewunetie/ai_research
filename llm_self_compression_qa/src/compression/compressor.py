"""Compressor module for creating self-compressed representations."""

from typing import Tuple, Optional
from ..models.base import BaseLLM
from .prompts import SELF_COMPRESSION_PROMPT, HUMAN_READABLE_SUMMARY_PROMPT
from .token_counter import TokenCounter


class Compressor:
    """
    Compressor role: creates compressed representations of documents.

    Supports both self-compression (non-human-readable) and human-readable
    summarization for baseline comparisons.
    """

    def __init__(self, model: BaseLLM, max_tokens: int = 1500):
        """
        Initialize Compressor.

        Args:
            model: LLM instance for compression
            max_tokens: Maximum tokens for compressed output
        """
        self.model = model
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model)

    def compress(
        self,
        text: str,
        compression_type: str = "self",
        **kwargs
    ) -> Tuple[str, dict]:
        """
        Compress text using specified compression method.

        Args:
            text: Original text to compress
            compression_type: "self" or "human_readable"
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (compressed_text, metadata_dict)

        Raises:
            ValueError: If compression_type is invalid
            RuntimeError: If compression fails
        """
        # Select prompt based on compression type
        if compression_type == "self":
            prompt = self._build_compression_prompt(text, SELF_COMPRESSION_PROMPT)
        elif compression_type == "human_readable":
            prompt = self._build_compression_prompt(text, HUMAN_READABLE_SUMMARY_PROMPT)
        else:
            raise ValueError(f"Invalid compression_type: {compression_type}")

        # Generate compressed version
        try:
            compressed = self.model.generate(prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Compression failed: {e}")

        # Count tokens
        original_tokens = self.token_counter.count(text)
        compressed_tokens = self.token_counter.count(compressed)

        # Check if within limit
        within_limit = compressed_tokens <= self.max_tokens

        # Metadata
        metadata = {
            "compression_type": compression_type,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": original_tokens / compressed_tokens if compressed_tokens > 0 else 0,
            "within_token_limit": within_limit,
            "token_limit": self.max_tokens,
            "model": self.model.get_model_name(),
        }

        # Warn if over limit
        if not within_limit:
            print(
                f"Warning: Compressed output ({compressed_tokens} tokens) "
                f"exceeds limit ({self.max_tokens} tokens)"
            )

        return compressed, metadata

    def _build_compression_prompt(self, text: str, compression_prompt: str) -> str:
        """
        Build full compression prompt.

        Args:
            text: Text to compress
            compression_prompt: Compression instruction prompt

        Returns:
            Complete prompt string
        """
        return f"{text}\n\n{compression_prompt}"

    def compress_batch(
        self,
        texts: list,
        compression_type: str = "self",
        **kwargs
    ) -> list:
        """
        Compress multiple texts.

        Args:
            texts: List of texts to compress
            compression_type: "self" or "human_readable"
            **kwargs: Additional generation parameters

        Returns:
            List of (compressed_text, metadata) tuples
        """
        results = []

        for i, text in enumerate(texts):
            print(f"Compressing document {i+1}/{len(texts)}...")
            compressed, metadata = self.compress(text, compression_type, **kwargs)
            results.append((compressed, metadata))

        return results

    def set_token_limit(self, max_tokens: int) -> None:
        """
        Update maximum token limit for compression.

        Args:
            max_tokens: New token limit
        """
        self.max_tokens = max_tokens
