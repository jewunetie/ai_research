"""Shared test fixtures and configuration for pytest."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_llm_model():
    """
    Create a mock LLM model with standard interface.

    Returns a mock that implements:
    - generate(prompt, **kwargs) -> str
    - count_tokens(text) -> int
    - get_model_name() -> str
    - encoding attribute with encode/decode methods
    """
    model = Mock()

    # Basic LLM interface
    model.generate.return_value = "Test response from LLM"
    model.count_tokens.side_effect = lambda text: len(text.split())
    model.get_model_name.return_value = "test-model"

    # Encoding interface (for token-level operations)
    mock_encoding = Mock()
    mock_encoding.encode.side_effect = lambda text: list(range(len(text.split())))
    mock_encoding.decode.side_effect = lambda tokens: " ".join([f"token{i}" for i in tokens])
    model.encoding = mock_encoding

    return model


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on
    developing algorithms that can learn from and make predictions or decisions
    based on data. Unlike traditional programming, where explicit instructions
    are coded, machine learning systems improve their performance through
    experience. Common applications include image recognition, natural language
    processing, and recommendation systems.
    """


@pytest.fixture
def sample_qa_pairs():
    """Sample QA pairs for testing."""
    return [
        {
            "question": "What is machine learning?",
            "reference_answer": "A subset of AI that focuses on developing algorithms that learn from data"
        },
        {
            "question": "What are common applications?",
            "reference_answer": "Image recognition, natural language processing, and recommendation systems"
        }
    ]


@pytest.fixture
def sample_compressed_text():
    """Sample compressed text for testing."""
    return "ML⊂AI→algos learn data→pred/dec≠trad prog→improv w/ exp→apps: img rec, NLP, recsys"
