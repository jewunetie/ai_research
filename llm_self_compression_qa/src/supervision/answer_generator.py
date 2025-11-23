"""Generate reference answers from documents for supervision."""

from typing import List, Dict, Any
import logging


ANSWER_GENERATION_PROMPT = """Answer the following question based ONLY on the provided context.
Provide a clear, concise answer using information directly from the context.

Context:
{context}

Question: {question}

Answer:"""


class AnswerGenerator:
    """
    Generate reference answers for questions from full documents.

    Used during supervision to create ground-truth answers for evaluation.
    """

    def __init__(self, model):
        """
        Initialize answer generator.

        Args:
            model: LLM model for generation
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 200,
    ) -> str:
        """
        Generate an answer to a question given context.

        Args:
            question: Question text
            context: Document or context text
            max_tokens: Maximum tokens in answer

        Returns:
            Generated answer string
        """
        prompt = ANSWER_GENERATION_PROMPT.format(
            context=context,
            question=question,
        )

        try:
            answer = self.model.generate(prompt, max_tokens=max_tokens)
            return answer.strip()

        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "ERROR"

    def generate_answers_batch(
        self,
        questions: List[str],
        context: str,
        max_tokens: int = 200,
    ) -> List[str]:
        """
        Generate answers for multiple questions with the same context.

        Args:
            questions: List of question strings
            context: Document or context text
            max_tokens: Maximum tokens per answer

        Returns:
            List of generated answers
        """
        answers = []

        for i, question in enumerate(questions):
            self.logger.debug(f"Generating answer {i + 1}/{len(questions)}")
            answer = self.generate_answer(question, context, max_tokens)
            answers.append(answer)

        return answers

    def update_qa_pairs_with_answers(
        self,
        qa_pairs: List[Dict[str, Any]],
        context: str,
        overwrite: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Add or update reference answers in QA pairs.

        Args:
            qa_pairs: List of QA pairs (must have 'question' field)
            context: Document context
            overwrite: If True, overwrite existing answers

        Returns:
            QA pairs with updated reference_answer fields
        """
        updated_pairs = []

        for qa in qa_pairs:
            qa_copy = qa.copy()

            # Generate answer if missing or overwrite requested
            if 'reference_answer' not in qa or overwrite:
                answer = self.generate_answer(qa['question'], context)
                qa_copy['reference_answer'] = answer

            updated_pairs.append(qa_copy)

        return updated_pairs
