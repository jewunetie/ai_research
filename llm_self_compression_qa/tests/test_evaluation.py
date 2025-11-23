"""Tests for evaluation module."""

import pytest
from unittest.mock import Mock
from src.evaluation.supervisor import Supervisor
from src.evaluation.answerer import Answerer


class TestSupervisor:
    """Test Supervisor (question generation)."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        # Mock JSON response with QA pairs
        model.generate.return_value = '''
[
    {"question": "What is the main topic?", "reference_answer": "Machine learning"},
    {"question": "Who is mentioned?", "reference_answer": "John Smith"}
]
'''
        return model

    @pytest.fixture
    def supervisor(self, mock_model):
        """Create supervisor with mock model."""
        return Supervisor(mock_model)

    def test_generate_questions_returns_list(self, supervisor):
        """Test that generate_questions returns a list."""
        text = "This is a test document about machine learning."

        qa_pairs = supervisor.generate_questions(text, num_questions=2)

        assert isinstance(qa_pairs, list)
        assert len(qa_pairs) > 0

    def test_generate_questions_structure(self, supervisor):
        """Test that QA pairs have correct structure."""
        text = "Test document"

        qa_pairs = supervisor.generate_questions(text, num_questions=2)

        for qa in qa_pairs:
            assert "question" in qa
            assert "reference_answer" in qa
            assert isinstance(qa["question"], str)
            assert isinstance(qa["reference_answer"], str)

    def test_generate_questions_calls_model(self, supervisor, mock_model):
        """Test that model.generate is called."""
        text = "Test document"

        supervisor.generate_questions(text)

        assert mock_model.generate.called
        # Check that text was passed in some form
        call_args = mock_model.generate.call_args[0][0]
        assert isinstance(call_args, str)

    def test_validate_qa_pair_valid(self, supervisor):
        """Test validation of valid QA pair."""
        valid_pair = {
            "question": "What is machine learning?",
            "reference_answer": "A field of AI"
        }

        assert supervisor.validate_qa_pair(valid_pair) is True

    def test_validate_qa_pair_missing_question(self, supervisor):
        """Test validation fails with missing question."""
        invalid_pair = {
            "reference_answer": "Some answer"
        }

        assert supervisor.validate_qa_pair(invalid_pair) is False

    def test_validate_qa_pair_missing_answer(self, supervisor):
        """Test validation fails with missing answer."""
        invalid_pair = {
            "question": "Some question?"
        }

        assert supervisor.validate_qa_pair(invalid_pair) is False

    def test_validate_qa_pair_empty_strings(self, supervisor):
        """Test validation fails with empty strings."""
        invalid_pair = {
            "question": "",
            "reference_answer": ""
        }

        assert supervisor.validate_qa_pair(invalid_pair) is False


class TestAnswerer:
    """Test Answerer."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.generate.return_value = "The answer is 42."
        return model

    @pytest.fixture
    def answerer(self, mock_model):
        """Create answerer with mock model."""
        return Answerer(mock_model)

    def test_answer_returns_string(self, answerer):
        """Test that answer returns a string."""
        context = "The answer to life is 42."
        question = "What is the answer to life?"

        answer = answerer.answer(context, question)

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_calls_model(self, answerer, mock_model):
        """Test that model.generate is called."""
        context = "Some context"
        question = "A question?"

        answerer.answer(context, question)

        assert mock_model.generate.called

    def test_answer_includes_context_and_question(self, answerer, mock_model):
        """Test that context and question are passed to model."""
        context = "Important context"
        question = "Test question?"

        answerer.answer(context, question)

        # Check that model was called with a prompt containing both
        call_args = mock_model.generate.call_args[0][0]
        assert isinstance(call_args, str)
        # The prompt should contain both context and question
        assert context in call_args or "context" in call_args.lower()

    def test_answer_strips_whitespace(self, answerer, mock_model):
        """Test that answer strips leading/trailing whitespace."""
        mock_model.generate.return_value = "  Answer with spaces  "

        answer = answerer.answer("context", "question")

        assert answer == "Answer with spaces"

    def test_answer_handles_unknown_response(self, answerer, mock_model):
        """Test handling of UNKNOWN responses."""
        mock_model.generate.return_value = "UNKNOWN"

        answer = answerer.answer("no relevant context", "unrelated question")

        assert "UNKNOWN" in answer.upper()


class TestSupervisionIntegration:
    """Integration tests for supervision pipeline."""

    @pytest.fixture
    def complete_mock_model(self):
        """Create complete mock model for integration tests."""
        model = Mock()

        # Mock different responses for different prompts
        def generate_side_effect(prompt, **kwargs):
            if "generate" in prompt.lower() and "question" in prompt.lower():
                # Question generation
                return '''[
                    {"question": "What is the main idea?", "reference_answer": "Self-compression"},
                    {"question": "How does it work?", "reference_answer": "Using LLMs"}
                ]'''
            else:
                # Question answering
                return "This is an answer to the question."

        model.generate.side_effect = generate_side_effect
        return model

    def test_full_qa_workflow(self, complete_mock_model):
        """Test complete QA workflow: generation + answering."""
        # Generate questions
        supervisor = Supervisor(complete_mock_model)
        text = "This paper discusses LLM self-compression."

        qa_pairs = supervisor.generate_questions(text, num_questions=2)

        assert len(qa_pairs) == 2

        # Answer questions
        answerer = Answerer(complete_mock_model)

        for qa in qa_pairs:
            answer = answerer.answer(text, qa["question"])

            assert isinstance(answer, str)
            assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
