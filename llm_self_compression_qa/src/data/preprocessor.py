"""Text preprocessing utilities for cleaning and normalizing documents."""

import re
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BaseLLM


class TextPreprocessor:
    """
    Preprocess and clean text for compression experiments.

    Handles:
    - Whitespace normalization
    - Special character cleaning
    - Token-aware truncation
    - Encoding issues
    """

    def __init__(self, model: Optional['BaseLLM'] = None):
        """
        Initialize preprocessor.

        Args:
            model: LLM model for token-aware operations (optional)
        """
        self.model = model

    def clean(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean
            aggressive: If True, apply more aggressive cleaning

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')

        # Normalize unicode
        text = self._normalize_unicode(text)

        # Fix common encoding issues
        text = self._fix_encoding_issues(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        if aggressive:
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)

            # Remove excessive punctuation
            text = re.sub(r'([.!?]){2,}', r'\1', text)

        # Final cleanup
        text = text.strip()

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Replace common unicode quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Replace em/en dashes with regular dash
        text = text.replace('—', '-').replace('–', '-')

        # Replace ellipsis
        text = text.replace('…', '...')

        return text

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix common mojibake patterns
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
        }

        for wrong, right in replacements.items():
            text = text.replace(wrong, right)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        add_truncation_notice: bool = False
    ) -> str:
        """
        Truncate text to maximum number of tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            add_truncation_notice: If True, add "[TRUNCATED]" marker

        Returns:
            Truncated text

        Raises:
            ValueError: If model not provided during initialization
        """
        if self.model is None:
            raise ValueError("Model required for token-aware truncation")

        # Count current tokens
        current_tokens = self.model.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Truncate using model's encoding
        encoding = self.model.encoding
        tokens = encoding.encode(text)

        # Reserve space for truncation notice if needed
        available_tokens = max_tokens - (5 if add_truncation_notice else 0)

        truncated_tokens = tokens[:available_tokens]
        truncated_text = encoding.decode(truncated_tokens)

        if add_truncation_notice:
            truncated_text += " [TRUNCATED]"

        return truncated_text

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
        preserve_sentences: bool = True
    ) -> list[str]:
        """
        Split text into chunks of approximately equal token count.

        Args:
            text: Text to split
            chunk_size: Target tokens per chunk
            overlap: Number of overlapping tokens between chunks
            preserve_sentences: Try to split on sentence boundaries

        Returns:
            List of text chunks

        Raises:
            ValueError: If model not provided during initialization
        """
        if self.model is None:
            raise ValueError("Model required for token-aware chunking")

        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

        encoding = self.model.encoding
        tokens = encoding.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk tokens
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)

            # Try to find sentence boundary if requested
            if preserve_sentences and end < len(tokens):
                # Look for sentence-ending punctuation near the end
                sentences_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

                # Search backwards from the end for a sentence boundary
                for i in range(len(chunk_text) - 1, max(len(chunk_text) - 100, 0), -1):
                    if any(chunk_text[i:i+2] == ending for ending in sentences_endings):
                        # Found a sentence boundary
                        chunk_text = chunk_text[:i+1]
                        # Recalculate token count
                        chunk_tokens = encoding.encode(chunk_text)
                        end = start + len(chunk_tokens)
                        break

            chunks.append(chunk_text.strip())

            # Move to next chunk with overlap
            start = end - overlap

        return chunks

    def remove_extra_formatting(self, text: str) -> str:
        """
        Remove excessive formatting that might interfere with compression.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple consecutive punctuation (except ...)
        text = re.sub(r'([!?]){2,}', r'\1', text)

        # Remove excessive capitalization (3+ caps in a row not at sentence start)
        def fix_caps(match):
            word = match.group(0)
            # Keep acronyms (all caps, no spaces)
            if word.isupper() and len(word) <= 5:
                return word
            # Otherwise lowercase
            return word.capitalize()

        text = re.sub(r'\b[A-Z]{3,}\b', fix_caps, text)

        return text

    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate text quality and return diagnostic information.

        Args:
            text: Text to validate

        Returns:
            Dictionary with validation results
        """
        if not text:
            return {
                "valid": False,
                "issues": ["Text is empty"],
                "length": 0,
                "word_count": 0,
            }

        issues = []

        # Check length
        if len(text) < 50:
            issues.append("Text is very short (< 50 characters)")

        # Check word count
        word_count = len(text.split())
        if word_count < 10:
            issues.append(f"Very few words ({word_count})")

        # Check for excessive non-ASCII
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii_ratio > 0.5:
            issues.append(f"High non-ASCII ratio ({non_ascii_ratio:.1%})")

        # Check for null bytes or control characters
        if '\x00' in text:
            issues.append("Contains null bytes")

        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 0:
            issues.append(f"Contains {control_chars} control characters")

        # Check for excessive whitespace
        whitespace_ratio = sum(1 for c in text if c.isspace()) / len(text)
        if whitespace_ratio > 0.5:
            issues.append(f"Excessive whitespace ({whitespace_ratio:.1%})")

        # Token count if model available
        token_count = None
        if self.model:
            token_count = self.model.count_tokens(text)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "length": len(text),
            "word_count": word_count,
            "token_count": token_count,
            "non_ascii_ratio": non_ascii_ratio,
        }
