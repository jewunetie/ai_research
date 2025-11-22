"""Answerer module for answering questions from compressed context."""

from typing import Dict, Any
from ..models.base import BaseLLM
from ..compression.prompts import QUESTION_ANSWERING_PROMPT


class Answerer:
    """
    Answerer role: answers questions using only the compressed representation.

    The Answerer does NOT have access to the original document, only the
    compressed version created by the Compressor.
    """

    def __init__(self, model: BaseLLM):
        """
        Initialize Answerer with an LLM.

        Args:
            model: LLM instance for answering questions
        """
        self.model = model

    def answer_question(
        self,
        question: str,
        compressed_context: str,
        **kwargs
    ) -> str:
        """
        Answer a question using compressed context.

        Args:
            question: Question to answer
            compressed_context: Compressed document representation
            **kwargs: Additional generation parameters

        Returns:
            Generated answer as string

        Raises:
            RuntimeError: If answer generation fails
        """
        # Build prompt with compressed context
        prompt = QUESTION_ANSWERING_PROMPT.format(
            context=compressed_context,
            question=question,
        )

        # Generate answer
        try:
            answer = self.model.generate(prompt, **kwargs)
            return answer.strip()
        except Exception as e:
            raise RuntimeError(f"Answer generation failed for question '{question}': {e}")

    def answer_multiple(
        self,
        questions: list,
        compressed_context: str,
        **kwargs
    ) -> list:
        """
        Answer multiple questions using the same compressed context.

        Args:
            questions: List of question strings or QA pair dicts
            compressed_context: Compressed document representation
            **kwargs: Additional generation parameters

        Returns:
            List of answer strings

        Raises:
            RuntimeError: If any answer generation fails
        """
        answers = []

        for q in questions:
            # Handle both string questions and QA pair dicts
            question_text = q if isinstance(q, str) else q.get("question", "")

            if not question_text:
                raise ValueError(f"Invalid question format: {q}")

            answer = self.answer_question(question_text, compressed_context, **kwargs)
            answers.append(answer)

        return answers

    def answer_with_metadata(
        self,
        question: str,
        compressed_context: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer question and return metadata about the process.

        Args:
            question: Question to answer
            compressed_context: Compressed document representation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with 'answer', 'question', and metadata
        """
        answer = self.answer_question(question, compressed_context, **kwargs)

        # Count tokens in context and answer
        context_tokens = self.model.count_tokens(compressed_context)
        answer_tokens = self.model.count_tokens(answer)

        return {
            "question": question,
            "answer": answer,
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
            "model": self.model.get_model_name(),
        }
