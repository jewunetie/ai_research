"""No context baseline - provides empty context."""


class NoContextBaseline:
    """
    Baseline that provides no context at all.

    This tests the lower bound: how well can the model answer
    questions with zero information about the document?
    """

    def process(self, text: str) -> tuple:
        """
        Return empty context.

        Args:
            text: Original text (unused)

        Returns:
            Tuple of (empty_string, metadata)
        """
        metadata = {
            "baseline_type": "no_context",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 0.0,
        }

        return "", metadata
