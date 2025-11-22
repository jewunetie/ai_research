"""Supervisor module for generating questions and reference answers."""

import json
from typing import List, Dict, Any
from ..models.base import BaseLLM
from ..compression.prompts import QUESTION_GENERATION_PROMPT


class Supervisor:
    """
    Supervisor role: generates questions and reference answers from original documents.

    The Supervisor has access to the full, uncompressed document and creates
    QA pairs that will be used to evaluate compression quality.
    """

    def __init__(self, model: BaseLLM):
        """
        Initialize Supervisor with an LLM.

        Args:
            model: LLM instance for question generation
        """
        self.model = model

    def generate_questions(
        self,
        text: str,
        num_questions: int = 10,
        question_types: List[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate questions and reference answers from text.

        Args:
            text: Original document text
            num_questions: Number of questions to generate
            question_types: Types of questions to include (not yet implemented)

        Returns:
            List of dicts with 'question' and 'reference_answer' fields

        Raises:
            RuntimeError: If generation or parsing fails
        """
        # Build prompt
        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            text=text,
        )

        # Generate questions
        try:
            response = self.model.generate(prompt)
        except Exception as e:
            raise RuntimeError(f"Question generation failed: {e}")

        # Parse response as JSON
        qa_pairs = self._parse_qa_response(response)

        # Validate we got the right number
        if len(qa_pairs) != num_questions:
            print(
                f"Warning: Requested {num_questions} questions, got {len(qa_pairs)}"
            )

        return qa_pairs

    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response into QA pairs.

        Tries JSON parsing first, falls back to heuristic parsing.

        Args:
            response: Raw LLM response

        Returns:
            List of QA pair dictionaries

        Raises:
            RuntimeError: If parsing fails completely
        """
        # Try to extract JSON from response
        try:
            # Look for JSON array in response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                qa_pairs = json.loads(json_str)

                # Validate structure
                if isinstance(qa_pairs, list) and all(
                    isinstance(item, dict)
                    and "question" in item
                    and "reference_answer" in item
                    for item in qa_pairs
                ):
                    return qa_pairs

        except json.JSONDecodeError:
            pass

        # Fallback: try parsing the entire response as JSON
        try:
            qa_pairs = json.loads(response)
            if isinstance(qa_pairs, list):
                return qa_pairs
        except json.JSONDecodeError:
            pass

        # If JSON parsing fails, raise error
        raise RuntimeError(
            f"Failed to parse question generation response. "
            f"Expected JSON list of objects with 'question' and 'reference_answer' fields. "
            f"Got: {response[:200]}..."
        )

    def validate_qa_pair(self, qa_pair: Dict[str, str]) -> bool:
        """
        Validate that a QA pair has required fields and reasonable content.

        Args:
            qa_pair: QA pair dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(qa_pair, dict):
            return False

        if "question" not in qa_pair or "reference_answer" not in qa_pair:
            return False

        # Check for non-empty strings
        if not qa_pair["question"].strip() or not qa_pair["reference_answer"].strip():
            return False

        return True
