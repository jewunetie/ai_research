"""QA pair validation utilities for supervision pipeline."""

from typing import Dict, List, Any, Tuple
import logging


class QAValidator:
    """
    Validate question-answer pairs for quality in supervision pipeline.

    This is a simplified interface that wraps the main QAValidator
    from src.data.validator for use in the supervision workflow.
    """

    def __init__(
        self,
        min_question_length: int = 10,
        max_question_length: int = 200,
        min_answer_length: int = 2,
        max_answer_length: int = 500,
        require_question_mark: bool = False,
    ):
        """
        Initialize QA validator.

        Args:
            min_question_length: Minimum characters in question
            max_question_length: Maximum characters in question
            min_answer_length: Minimum characters in answer
            max_answer_length: Maximum characters in answer
            require_question_mark: If True, questions must end with '?'
        """
        # Import the main validator
        from src.data.validator import QAValidator as MainQAValidator

        self.validator = MainQAValidator(
            min_question_length=min_question_length,
            max_question_length=max_question_length,
            min_answer_length=min_answer_length,
            max_answer_length=max_answer_length,
            require_question_mark=require_question_mark,
        )
        self.logger = logging.getLogger(__name__)

    def validate_qa_pair(self, qa: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single QA pair.

        Args:
            qa: Dictionary with 'question' and 'answer' or 'reference_answer' fields

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        return self.validator.validate_qa_pair(qa)

    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a list of QA pairs.

        Args:
            qa_pairs: List of QA pair dictionaries

        Returns:
            Dictionary with validation summary
        """
        result = self.validator.validate_qa_pairs(qa_pairs)

        # Log summary
        self.logger.info(
            f"QA Validation: {result['valid']}/{result['total']} valid "
            f"({result['valid_ratio']:.1%})"
        )

        if result['invalid'] > 0:
            self.logger.warning(f"Found {result['invalid']} invalid QA pairs")
            if result['common_issues']:
                self.logger.warning(f"Common issues: {result['common_issues']}")

        return result

    def filter_valid_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter QA pairs to return only valid ones.

        Args:
            qa_pairs: List of QA pair dictionaries

        Returns:
            List of valid QA pairs
        """
        valid_pairs = self.validator.filter_valid_pairs(qa_pairs)

        filtered_count = len(qa_pairs) - len(valid_pairs)
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} invalid QA pairs")

        return valid_pairs

    @staticmethod
    def validate_qa_pair_simple(qa: Dict) -> bool:
        """
        Simple validation check for QA pair.

        Args:
            qa: QA pair dictionary

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if 'question' not in qa:
            return False

        answer_field = 'reference_answer' if 'reference_answer' in qa else 'answer'
        if answer_field not in qa:
            return False

        question = qa['question']
        answer = qa[answer_field]

        # Basic checks
        if not question or not question.strip():
            return False

        if not answer or not answer.strip():
            return False

        # Question must be non-empty and reasonable length
        if len(question.strip()) < 10:
            return False

        # Answer must have substance (>2 tokens)
        if len(answer.strip().split()) < 2:
            return False

        return True

    @staticmethod
    def filter_qa_pairs_simple(qa_pairs: List[Dict]) -> List[Dict]:
        """
        Filter QA pairs using simple validation.

        Args:
            qa_pairs: List of QA pairs

        Returns:
            List of valid QA pairs
        """
        return [qa for qa in qa_pairs if QAValidator.validate_qa_pair_simple(qa)]
