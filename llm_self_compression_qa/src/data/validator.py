"""Data validation utilities for ensuring quality of documents and QA pairs."""

from typing import Dict, List, Any, Tuple, Optional
import re


class DocumentValidator:
    """
    Validate document quality for compression experiments.

    Ensures documents meet minimum requirements for meaningful compression
    and question answering.
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 100000,
        min_words: int = 20,
        max_non_ascii_ratio: float = 0.3,
    ):
        """
        Initialize validator with quality thresholds.

        Args:
            min_length: Minimum character count
            max_length: Maximum character count
            min_words: Minimum word count
            max_non_ascii_ratio: Maximum ratio of non-ASCII characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_non_ascii_ratio = max_non_ascii_ratio

    def validate_document(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single document.

        Args:
            document: Document dictionary with 'text' field

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        if 'text' not in document:
            return False, ["Missing 'text' field"]

        text = document['text']

        # Check for empty or None
        if not text or not text.strip():
            return False, ["Text is empty or whitespace-only"]

        # Length checks
        if len(text) < self.min_length:
            issues.append(f"Text too short: {len(text)} < {self.min_length} characters")

        if len(text) > self.max_length:
            issues.append(f"Text too long: {len(text)} > {self.max_length} characters")

        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            issues.append(f"Too few words: {len(words)} < {self.min_words}")

        # Check for meaningful content (not just numbers/punctuation)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.5:
            issues.append(f"Low alphabetic content: {alpha_ratio:.1%}")

        # Check non-ASCII ratio
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / len(text)
        if non_ascii_ratio > self.max_non_ascii_ratio:
            issues.append(f"High non-ASCII ratio: {non_ascii_ratio:.1%} > {self.max_non_ascii_ratio:.1%}")

        # Check for problematic characters
        if '\x00' in text:
            issues.append("Contains null bytes")

        # Check for excessive repetition (might indicate corrupted data)
        if self._has_excessive_repetition(text):
            issues.append("Excessive character/word repetition detected")

        return len(issues) == 0, issues

    def _has_excessive_repetition(self, text: str, threshold: int = 50) -> bool:
        """
        Check if text has excessive repetition (likely corrupted data).

        Args:
            text: Text to check
            threshold: Maximum allowed repetitions

        Returns:
            True if excessive repetition detected
        """
        # Check for repeated characters
        if re.search(r'(.)\1{' + str(threshold) + r',}', text):
            return True

        # Check for repeated words
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

            # If any word appears more than 20% of the time, flag it
            max_count = max(word_counts.values())
            if max_count / len(words) > 0.2:
                return True

        return False

    def validate_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of documents.

        Args:
            documents: List of document dictionaries

        Returns:
            Dictionary with validation summary
        """
        valid_docs = []
        invalid_docs = []
        all_issues = []

        for i, doc in enumerate(documents):
            is_valid, issues = self.validate_document(doc)

            if is_valid:
                valid_docs.append(doc)
            else:
                invalid_docs.append({
                    'index': i,
                    'doc_id': doc.get('id', f'doc_{i}'),
                    'issues': issues,
                })
                all_issues.extend(issues)

        return {
            'total': len(documents),
            'valid': len(valid_docs),
            'invalid': len(invalid_docs),
            'valid_ratio': len(valid_docs) / len(documents) if documents else 0,
            'valid_documents': valid_docs,
            'invalid_documents': invalid_docs,
            'common_issues': self._get_common_issues(all_issues),
        }

    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Get frequency of different issue types."""
        issue_counts = {}
        for issue in issues:
            # Categorize issue by first few words
            category = ' '.join(issue.split()[:3])
            issue_counts[category] = issue_counts.get(category, 0) + 1

        # Sort by frequency
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))


class QAValidator:
    """
    Validate question-answer pairs for quality.

    Ensures QA pairs are suitable for evaluation.
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
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.require_question_mark = require_question_mark

    def validate_qa_pair(self, qa: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single QA pair.

        Args:
            qa: Dictionary with 'question' and 'answer' or 'reference_answer' fields

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        if 'question' not in qa:
            return False, ["Missing 'question' field"]

        answer_field = 'reference_answer' if 'reference_answer' in qa else 'answer'
        if answer_field not in qa:
            return False, ["Missing answer field"]

        question = qa['question']
        answer = qa[answer_field]

        # Validate question
        if not question or not question.strip():
            issues.append("Question is empty")
        else:
            # Length checks
            q_len = len(question.strip())
            if q_len < self.min_question_length:
                issues.append(f"Question too short: {q_len} < {self.min_question_length} chars")

            if q_len > self.max_question_length:
                issues.append(f"Question too long: {q_len} > {self.max_question_length} chars")

            # Question mark check
            if self.require_question_mark and not question.strip().endswith('?'):
                issues.append("Question does not end with '?'")

            # Check for question words
            question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'is', 'are', 'do', 'does']
            has_question_word = any(question.lower().startswith(word) for word in question_words)
            if not has_question_word and not question.strip().endswith('?'):
                issues.append("Question doesn't start with question word and doesn't end with '?'")

        # Validate answer
        if not answer or not answer.strip():
            issues.append("Answer is empty")
        else:
            # Length checks
            a_len = len(answer.strip())
            if a_len < self.min_answer_length:
                issues.append(f"Answer too short: {a_len} < {self.min_answer_length} chars")

            if a_len > self.max_answer_length:
                issues.append(f"Answer too long: {a_len} > {self.max_answer_length} chars")

            # Word count check (answers should have substance)
            answer_words = answer.strip().split()
            if len(answer_words) < 2:
                issues.append(f"Answer has too few words: {len(answer_words)}")

        # Check for duplicate question/answer
        if question and answer and question.strip().lower() == answer.strip().lower():
            issues.append("Question and answer are identical")

        return len(issues) == 0, issues

    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a list of QA pairs.

        Args:
            qa_pairs: List of QA pair dictionaries

        Returns:
            Dictionary with validation summary
        """
        valid_pairs = []
        invalid_pairs = []
        all_issues = []

        for i, qa in enumerate(qa_pairs):
            is_valid, issues = self.validate_qa_pair(qa)

            if is_valid:
                valid_pairs.append(qa)
            else:
                invalid_pairs.append({
                    'index': i,
                    'qa_id': qa.get('question_id', i),
                    'question': qa.get('question', ''),
                    'issues': issues,
                })
                all_issues.extend(issues)

        return {
            'total': len(qa_pairs),
            'valid': len(valid_pairs),
            'invalid': len(invalid_pairs),
            'valid_ratio': len(valid_pairs) / len(qa_pairs) if qa_pairs else 0,
            'valid_pairs': valid_pairs,
            'invalid_pairs': invalid_pairs,
            'common_issues': self._get_common_issues(all_issues),
        }

    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Get frequency of different issue types."""
        issue_counts = {}
        for issue in issues:
            # Categorize issue by first few words
            category = ' '.join(issue.split()[:3])
            issue_counts[category] = issue_counts.get(category, 0) + 1

        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))

    def filter_valid_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter QA pairs to return only valid ones.

        Args:
            qa_pairs: List of QA pair dictionaries

        Returns:
            List of valid QA pairs
        """
        return [qa for qa in qa_pairs if self.validate_qa_pair(qa)[0]]
