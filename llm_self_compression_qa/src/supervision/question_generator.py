"""Question generation from documents for supervision."""

from typing import List, Dict, Any
import re
import logging


# Question generation prompt template
QA_GENERATION_PROMPT = """Given the following document, generate exactly {num_questions} questions that test comprehension:
- {factual_count} factual questions about who, what, when, where
- {detail_count} detail-oriented questions about specific facts, numbers, locations
- {inferential_count} inferential questions about why, how, consequences
- {conceptual_count} conceptual questions about main ideas, themes, relationships

For each question, also provide the answer based ONLY on the document.

Document:
{text}

Output format (use exactly this format):
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
Q{num_questions}: [question]
A{num_questions}: [answer]
"""


class QuestionGenerator:
    """
    Generate questions and reference answers from documents.

    Uses an LLM to create diverse questions that test different aspects
    of document comprehension.
    """

    def __init__(
        self,
        model,
        num_questions: int = 10,
        question_distribution: Dict[str, int] = None,
    ):
        """
        Initialize question generator.

        Args:
            model: LLM model for generation
            num_questions: Number of questions to generate per document
            question_distribution: Distribution of question types
                (default: {factual: 3, detail: 3, inferential: 2, conceptual: 2})
        """
        self.model = model
        self.num_questions = num_questions
        self.logger = logging.getLogger(__name__)

        # Default distribution
        if question_distribution is None:
            self.question_distribution = {
                'factual': 3,
                'detail': 3,
                'inferential': 2,
                'conceptual': 2,
            }
        else:
            self.question_distribution = question_distribution

    def generate_qa_pairs(
        self,
        document: str,
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions and answers from a document.

        Args:
            document: Document text
            max_retries: Maximum number of retries if parsing fails

        Returns:
            List of QA pair dictionaries with fields:
                - question_id: int
                - question: str
                - reference_answer: str

        Raises:
            RuntimeError: If generation fails after max_retries
        """
        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"Generating QA pairs (attempt {attempt + 1}/{max_retries + 1})")

                # Create prompt
                prompt = QA_GENERATION_PROMPT.format(
                    num_questions=self.num_questions,
                    factual_count=self.question_distribution['factual'],
                    detail_count=self.question_distribution['detail'],
                    inferential_count=self.question_distribution['inferential'],
                    conceptual_count=self.question_distribution['conceptual'],
                    text=document,
                )

                # Generate
                response = self.model.generate(prompt, max_tokens=2000)

                # Parse
                qa_pairs = self._parse_qa_response(response)

                if len(qa_pairs) >= self.num_questions * 0.7:  # Accept if got at least 70%
                    self.logger.info(f"Generated {len(qa_pairs)} QA pairs")
                    return qa_pairs
                else:
                    self.logger.warning(
                        f"Only parsed {len(qa_pairs)}/{self.num_questions} QA pairs. Retrying..."
                    )

            except Exception as e:
                self.logger.error(f"Error generating QA pairs: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to generate QA pairs after {max_retries + 1} attempts") from e

        return []

    def _parse_qa_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse structured Q/A response.

        Expected format:
            Q1: [question]
            A1: [answer]
            Q2: [question]
            A2: [answer]
            ...

        Args:
            response: Raw model response

        Returns:
            List of parsed QA pairs
        """
        qa_pairs = []

        # Pattern: Q1: ... A1: ... Q2: ... A2: ...
        # Use non-greedy matching and lookahead
        pattern = r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)'

        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            question_num = match.group(1)
            question = match.group(2).strip()
            answer = match.group(3).strip()

            if question and answer:  # Only add if both non-empty
                qa_pairs.append({
                    "question_id": int(question_num),
                    "question": question,
                    "reference_answer": answer,
                })

        return qa_pairs

    def generate_with_validation(
        self,
        document: str,
        validator=None,
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs with optional validation.

        Args:
            document: Document text
            validator: QAValidator instance (optional)
            max_retries: Maximum retries

        Returns:
            List of valid QA pairs
        """
        qa_pairs = self.generate_qa_pairs(document, max_retries=max_retries)

        if validator is not None:
            # Filter to valid pairs only
            qa_pairs = validator.filter_valid_pairs(qa_pairs)
            self.logger.info(f"After validation: {len(qa_pairs)} valid pairs")

        return qa_pairs
