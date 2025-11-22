# Implementation Plan: LLM Self-Compression for QA

## Overview

This document provides a detailed, phased implementation plan for testing LLM self-compression capabilities through downstream question answering.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  Document Loader  →  Preprocessor  →  Data Validator            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Compression Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Compressor LLM  →  Token Counter  →  Compliance Checker        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Supervision Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Question Generator  →  Answer Generator  →  QA Validator       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Answerer LLM  →  Metric Calculator  →  Result Aggregator       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis & Reporting                          │
├─────────────────────────────────────────────────────────────────┤
│  Statistical Analysis  →  Visualization  →  Report Generator    │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
llm_self_compression_qa/
├── CLAUDE.md                          # Project overview
├── RESEARCH.md                        # Literature review
├── IMPLEMENTATION.md                  # This file
├── README.md                          # Setup and usage instructions
│
├── pyproject.toml                     # uv project configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract LLM interface
│   │   ├── openai_model.py            # OpenAI API wrapper
│   │   ├── anthropic_model.py         # Anthropic API wrapper
│   │   ├── huggingface_model.py       # HF models (Llama, Mistral)
│   │   └── model_factory.py           # Model instantiation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Load datasets (CNN/DailyMail, etc.)
│   │   ├── preprocessor.py            # Clean and normalize text
│   │   └── validator.py               # Data quality checks
│   │
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── compressor.py              # Compression logic
│   │   ├── prompts.py                 # Compression prompt templates
│   │   └── token_counter.py           # Token counting and enforcement
│   │
│   ├── supervision/
│   │   ├── __init__.py
│   │   ├── question_generator.py      # Generate questions from documents
│   │   ├── answer_generator.py        # Generate reference answers
│   │   └── qa_validator.py            # QA quality control
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── answerer.py                # Answer questions from compressed form
│   │   ├── metrics.py                 # EM, F1, semantic similarity
│   │   └── aggregator.py              # Aggregate results across experiments
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── full_context.py            # Full-context QA baseline
│   │   ├── human_summary.py           # Human-readable summary baseline
│   │   ├── random_sample.py           # Random token sampling
│   │   └── llmlingua_style.py         # Smart token selection (optional)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py              # Statistical tests, effect sizes
│   │   ├── visualization.py           # Plots and charts
│   │   └── report_generator.py        # Generate markdown/HTML reports
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging_config.py          # Logging setup
│       └── helpers.py                 # Utility functions
│
├── experiments/
│   ├── pilot/                         # Pilot experiment scripts
│   │   ├── run_pilot.py
│   │   └── analyze_pilot.py
│   │
│   ├── main/                          # Main experiment scripts
│   │   ├── run_main.py
│   │   └── analyze_main.py
│   │
│   └── configs/                       # Experiment configurations
│       ├── pilot_config.yaml
│       └── main_config.yaml
│
├── data/                              # Data storage
│   ├── raw/                           # Original datasets
│   ├── processed/                     # Preprocessed documents
│   └── generated/                     # Generated compressions and QA pairs
│
├── results/                           # Experimental results
│   ├── pilot/                         # Pilot results
│   ├── main/                          # Main experiment results
│   └── figures/                       # Generated plots
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_compression_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
└── tests/                             # Unit tests
    ├── test_compression.py
    ├── test_evaluation.py
    └── test_metrics.py
```

---

## 🚨 CRITICAL: 2025 API Updates

**This implementation uses the latest OpenAI APIs as of November 2025:**

**⚠️ IMPORTANT: VERIFICATION REQUIRED DURING PHASE 0**

Since OpenAI's official documentation was inaccessible during planning, the following MUST be verified when you begin implementation:

1. **Responses API Syntax**: Confirm `client.responses.create()` exists and uses `input` parameter
2. **Response Format**: Verify response has `output_text` attribute (fallback to `choices` included)
3. **Model Names**: Test actual model names (`gpt-5.1-chat-latest`, `gpt-5.1-thinking`, etc.)
4. **Parameters**: Confirm `seed`, `reasoning_effort`, `max_tokens` are supported
5. **Pricing**: Check actual GPT-5.1 costs before running experiments (pricing TBD as of Nov 2025)

**Fallback Strategy**: If Responses API doesn't exist, code includes graceful fallback to Chat Completions API format.

### Major Changes from Pre-2025 Implementations:

1. **Responses API** (introduced March 2025):
   - **NEW**: `client.responses.create(model=..., input=...)`
   - **OLD**: `client.chat.completions.create(model=..., messages=...)`
   - Responses API is the recommended approach for new projects
   - Chat Completions API still supported but considered legacy

2. **GPT-5 Series Models** (released August 2025):
   - GPT-5: State-of-the-art base model
   - GPT-5.1 Instant (`gpt-5.1-chat-latest`): Fast, adaptive reasoning
   - GPT-5.1 Thinking (`gpt-5.1-thinking`): Deep reasoning
   - Significant performance improvements over GPT-4

3. **openai-python v2.0+** (2025):
   - Breaking changes in v2.0.0
   - MUST use `openai>=2.0.0` for Responses API
   - ResponseFunctionToolCallOutputItem.output returns Array, not just string

4. **New Parameters**:
   - `reasoning_effort`: Control reasoning depth ("none", "minimal", "medium", "high")
   - `seed`: Improved reproducibility support
   - `input`: Replaces `messages` in Responses API

### Why This Matters for Our Research:

- **Better compression**: GPT-5.1 may create more sophisticated compression schemes
- **Reproducibility**: `seed` parameter ensures consistent results across runs
- **Adaptive reasoning**: Can enable deeper thinking when needed for complex compressions

---

## Implementation Phases

### Phase 0: Setup (Estimated: 2-4 hours)

**Goals**:
- Set up development environment
- Install dependencies
- Configure API keys
- Verify model access

**Tasks**:
1. Initialize project with `uv` or pip
2. Create virtual environment
3. Create `pyproject.toml` for package installation
4. Install core dependencies:
   - `openai` for GPT models
   - `anthropic` for Claude models
   - `transformers` for HF models (optional)
   - `tiktoken` for token counting
   - `sentence-transformers` for semantic similarity
   - `datasets` for loading benchmarks
   - `pandas`, `numpy` for data manipulation
   - `matplotlib`, `seaborn` for visualization
   - `pytest` for testing
   - `pyyaml` for configuration
4. Set up `.env` file with API keys
5. Install package in development mode: `pip install -e .`
6. Test API connectivity

**Example pyproject.toml** (Updated for 2025):
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-self-compression-qa"
version = "0.1.0"
description = "LLM Self-Compression for Question Answering Research"
requires-python = ">=3.10,<3.13"  # Restrict due to tiktoken compatibility issues
dependencies = [
    "openai>=2.0.0",  # CRITICAL: v2.0+ required for Responses API (March 2025)
    "tiktoken>=0.7.0",  # Latest version for GPT-5 models
    "sentence-transformers>=3.0.0",
    "datasets",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "flake8",
]
```

**IMPORTANT COMPATIBILITY NOTES:**

1. **OpenAI Library Version**:
   - **MUST use v2.0.0 or higher** for Responses API support
   - v2.0.0 introduced in 2025 with breaking changes
   - Chat Completions API still works but Responses API is recommended

2. **Python Version**:
   - **Recommended: Python 3.10 or 3.11**
   - Python 3.12 has known tiktoken compatibility issues (as of Nov 2025)
   - Test tiktoken installation before proceeding with 3.12+

3. **tiktoken Encoding**:
   - GPT-5/5.1 models use `o200k_base` encoding
   - Older tiktoken versions may not recognize new model names
   - Implementation includes fallback to `o200k_base`

**Deliverables**:
- Working Python environment
- Configuration files (pyproject.toml, .env)
- README.md with setup instructions

**Acceptance Criteria**:
- Can install package with `pip install -e .`
- Can import from src package
- Can load a model and generate a completion
- Can count tokens accurately
- All tests pass

**CRITICAL VERIFICATION STEPS** (Phase 0):

```python
# Test script to verify API works
from openai import OpenAI

client = OpenAI()

# Test 1: Verify Responses API exists
try:
    response = client.responses.create(
        model="gpt-5.1-chat-latest",  # or "gpt-4o" as fallback
        input="Hello, this is a test.",
        temperature=0.0,
        seed=42
    )
    print("✅ Responses API works!")
    print(f"Response: {response.output_text}")
except AttributeError:
    print("❌ Responses API not found. Use Chat Completions instead:")
    response = client.chat.completions.create(
        model="gpt-4o",  # Use available model
        messages=[{"role": "user", "content": "Hello, this is a test."}],
        temperature=0.0,
        seed=42
    )
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check model name and API key")
```

Run this test immediately after installation to determine which API version to use.

---

### Phase 1: Core Infrastructure (Estimated: 1 day)

**Goals**:
- Implement model interfaces
- Implement token counting
- Implement basic compression pipeline
- Set up logging

**Tasks**:

**1.1: Model Abstraction Layer**
```python
# src/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier."""
        pass
```

**1.2: OpenAI Implementation (Updated for 2025 Responses API)**
```python
# src/models/openai_model.py
from openai import OpenAI
import tiktoken
from .base import BaseLLM

class OpenAIModel(BaseLLM):
    """
    OpenAI model wrapper using the Responses API (introduced March 2025).

    Recommended models (as of November 2025):
    - gpt-5.1-chat-latest (GPT-5.1 Instant - fast, adaptive reasoning)
    - gpt-5.1 or gpt-5.1-thinking (GPT-5.1 Thinking - deep reasoning)
    - gpt-5 (base GPT-5 model)
    - gpt-4o (GPT-4 optimized, fallback option)
    """

    def __init__(
        self,
        model_name="gpt-5.1-chat-latest",  # Updated default to GPT-5.1 Instant
        temperature=0.0,
        seed=42,  # For reproducibility
        reasoning_effort="none",  # "none", "minimal", "medium", "high" for GPT-5.1
        api_key=None
    ):
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if api_key=None
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.reasoning_effort = reasoning_effort

        # Get encoding (may need fallback for newer models)
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for new models not yet in tiktoken
            self.encoding = tiktoken.get_encoding("o200k_base")  # GPT-4o/5 encoding

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate completion using the Responses API (March 2025+).

        Note: Responses API uses 'input' parameter instead of 'messages'.

        Raises:
            RuntimeError: If API call fails or response format is unexpected
        """
        # Prepare parameters for Responses API
        params = {
            "model": self.model_name,
            "input": prompt,  # Responses API uses 'input' not 'messages'
            "temperature": self.temperature,
            "seed": self.seed,
        }

        # Add reasoning_effort for GPT-5.1 models
        if "5.1" in self.model_name or "gpt-5" in self.model_name:
            params["reasoning_effort"] = self.reasoning_effort

        # Merge with any additional kwargs
        params.update(kwargs)

        try:
            # Call Responses API
            response = self.client.responses.create(**params)

            # Extract text from response
            # Responses API returns output_text or structured output
            if hasattr(response, 'output_text'):
                return response.output_text
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                # Fallback to Chat Completions format if needed
                return response.choices[0].message.content
            else:
                raise RuntimeError(f"Unexpected response format: {response}")

        except AttributeError as e:
            raise RuntimeError(
                f"Response missing expected attributes. API may have changed. "
                f"Error: {e}. Response: {response if 'response' in locals() else 'N/A'}"
            )
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def get_model_name(self) -> str:
        return self.model_name
```

**Alternative: Chat Completions API Implementation (Fallback)**

If Responses API is not available, use this implementation instead:

```python
# src/models/openai_model.py (Chat Completions version)
from openai import OpenAI
import tiktoken
from .base import BaseLLM

class OpenAIModel(BaseLLM):
    """
    OpenAI model wrapper using Chat Completions API (legacy but stable).
    Use this if Responses API is not available.
    """

    def __init__(
        self,
        model_name="gpt-4o",  # Verified working model
        temperature=0.0,
        seed=42,
        api_key=None
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion using Chat Completions API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                seed=self.seed,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def get_model_name(self) -> str:
        return self.model_name
```

**1.3: Token Counter**
```python
# src/compression/token_counter.py
from typing import Tuple

class TokenCounter:
    def __init__(self, model):
        self.model = model

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return self.model.count_tokens(text)

    def enforce_limit(self, text: str, limit: int) -> Tuple[str, int, bool]:
        """
        Enforce token limit on text.

        Returns:
            (text, token_count, compliant)
        """
        token_count = self.count(text)
        compliant = token_count <= limit

        if compliant:
            return text, token_count, True

        # Truncate to limit using proper token-based truncation
        encoding = self.model.encoding
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:limit]
        truncated_text = encoding.decode(truncated_tokens)

        return truncated_text, limit, False
```

**1.4: Logging Setup**
```python
# src/utils/logging_config.py
import logging
from pathlib import Path

def setup_logging(log_dir: Path, experiment_name: str):
    log_file = log_dir / f"{experiment_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(experiment_name)
```

**Deliverables**:
- Working model interfaces for OpenAI (minimum)
- Token counting functionality
- Logging system
- Unit tests for core functionality

**Acceptance Criteria**:
- Can instantiate model and generate text
- Can accurately count tokens
- All logs are captured
- Tests pass

---

### Phase 2: Data Pipeline (Estimated: 0.5 days)

**Goals**:
- Load datasets
- Preprocess documents
- Validate data quality

**Tasks**:

**2.1: Dataset Loader**
```python
# src/data/loader.py
from datasets import load_dataset
from typing import List, Dict

class DatasetLoader:
    @staticmethod
    def load_cnn_dailymail(split="test", num_samples=100) -> List[Dict]:
        """Load CNN/DailyMail dataset."""
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

        documents = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            documents.append({
                "id": f"cnn_dm_{i}",
                "text": item["article"],
                "highlights": item["highlights"],  # Can use for validation
                "source": "cnn_dailymail"
            })

        return documents

    @staticmethod
    def load_wikipedia(topics: List[str], num_per_topic=10) -> List[Dict]:
        """Load Wikipedia articles."""
        # Implementation depends on Wikipedia API or dataset
        pass
```

**2.2: Text Preprocessor**
```python
# src/data/preprocessor.py
import re

class TextPreprocessor:
    @staticmethod
    def clean(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters that might break tokenization
        text = text.strip()

        return text

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int, model) -> str:
        """Truncate text to maximum number of tokens."""
        # Implementation with proper token-aware truncation
        pass
```

**Deliverables**:
- Dataset loading for CNN/DailyMail
- Text preprocessing utilities
- Data validation checks

**Acceptance Criteria**:
- Can load 100 CNN/DailyMail articles
- Text is properly cleaned
- Token counts are correct

---

### Phase 3: Compression Pipeline (Estimated: 0.5 days)

**Goals**:
- Implement compression with various prompts
- Enforce token limits
- Store compressed outputs

**Tasks**:

**3.1: Compression Prompts**
```python
# src/compression/prompts.py

SELF_COMPRESSION_PROMPT = """
Now summarize everything in this text in as much detail as possible, but compress it as much as possible into a format that you can still read. It does not need to be human readable. You do not need to use a common character set, all that matters is we can pick back up right where we left off if I were to start a new conversation with you. You are limited to {token_limit} tokens.

Text to compress:
{document}

Compressed representation:
"""

HUMAN_SUMMARY_PROMPT = """
Summarize the following text in {token_limit} tokens or less, preserving key information in a clear, human-readable format.

Text to summarize:
{document}

Summary:
"""

ENHANCED_SELF_COMPRESSION_PROMPT = """
Compress the following text into at most {token_limit} tokens using ANY encoding scheme you want.

IMPORTANT:
- Do NOT write a standard summary
- Use compact notation, abbreviations, symbols, or any encoding that maximizes information density
- Examples: "J.S.|45|tchr|NYC" instead of "John Smith, 45 years old, teacher from New York City"
- Invent your own notation if helpful
- Only requirement: you must be able to answer questions from this compressed form later

Text to compress:
{document}

Compressed representation:
"""

# Prompt registry for configuration files
PROMPT_REGISTRY = {
    "SELF_COMPRESSION_PROMPT": SELF_COMPRESSION_PROMPT,
    "HUMAN_SUMMARY_PROMPT": HUMAN_SUMMARY_PROMPT,
    "ENHANCED_SELF_COMPRESSION_PROMPT": ENHANCED_SELF_COMPRESSION_PROMPT,
}

def get_prompt(name: str) -> str:
    """Get prompt template by name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]
```

**3.2: Compressor**
```python
# src/compression/compressor.py
from typing import Dict, Any
import logging
from .token_counter import TokenCounter

class Compressor:
    def __init__(self, model):
        self.model = model
        self.token_counter = TokenCounter(model)
        self.logger = logging.getLogger(__name__)

    def compress(
        self,
        document: str,
        prompt_template: str,
        token_limit: int = 1500
    ) -> Dict[str, Any]:
        """
        Compress document using specified prompt.

        Returns:
            {
                "compressed": str,
                "token_count": int,
                "compliant": bool,
                "original_tokens": int,
                "compression_ratio": float
            }
        """
        # Create prompt
        prompt = prompt_template.format(
            document=document,
            token_limit=token_limit
        )

        # Generate compression
        self.logger.info(f"Compressing document (target: {token_limit} tokens)")
        compressed = self.model.generate(prompt)

        # Count tokens and check compliance
        compressed_clean, token_count, compliant = self.token_counter.enforce_limit(
            compressed, token_limit
        )

        # Calculate statistics
        original_tokens = self.token_counter.count(document)
        compression_ratio = original_tokens / token_count if token_count > 0 else 0

        result = {
            "compressed": compressed_clean,
            "token_count": token_count,
            "compliant": compliant,
            "original_tokens": original_tokens,
            "compression_ratio": compression_ratio
        }

        self.logger.info(
            f"Compression complete: {original_tokens} → {token_count} tokens "
            f"({compression_ratio:.2f}x compression)"
        )

        return result
```

**Deliverables**:
- Compression pipeline with multiple prompt variants
- Token enforcement
- Compression statistics

**Acceptance Criteria**:
- Can compress documents with different prompts
- Token limits are enforced
- Compression ratios are calculated correctly

---

### Phase 4: Question Generation (Estimated: 0.5 days)

**Goals**:
- Generate diverse questions from documents
- Generate reference answers
- Validate QA quality

**Tasks**:

**4.1: Question Generator**
```python
# src/supervision/question_generator.py
from typing import List, Dict
import re

QA_GENERATION_PROMPT = """
Given the following document, generate exactly 10 questions that test comprehension:
- 3 factual questions about who, what, when, where
- 3 detail-oriented questions about specific facts, numbers, locations
- 2 inferential questions about why, how, consequences
- 2 conceptual questions about main ideas, themes, relationships

For each question, also provide the answer based ONLY on the document.

Document:
{document}

Output format (use exactly this format):
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
Q10: [question]
A10: [answer]
"""

class QuestionGenerator:
    def __init__(self, model):
        self.model = model

    def generate_qa_pairs(self, document: str) -> List[Dict[str, str]]:
        """Generate questions and answers from document."""
        prompt = QA_GENERATION_PROMPT.format(document=document)
        response = self.model.generate(prompt, max_tokens=2000)

        # Parse Q/A pairs
        qa_pairs = self._parse_qa_response(response)

        return qa_pairs

    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """Parse structured Q/A response."""
        qa_pairs = []

        # Pattern: Q1: ... A1: ... Q2: ... A2: ...
        pattern = r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)'
        matches = re.finditer(pattern, response, re.DOTALL)

        for match in matches:
            question_num = match.group(1)
            question = match.group(2).strip()
            answer = match.group(3).strip()

            qa_pairs.append({
                "question_id": int(question_num),
                "question": question,
                "reference_answer": answer
            })

        return qa_pairs
```

**4.2: QA Validator**
```python
# src/supervision/qa_validator.py

class QAValidator:
    @staticmethod
    def validate_qa_pair(qa: Dict) -> bool:
        """Validate single QA pair."""
        # Question must be non-empty and end with question mark
        if not qa["question"] or len(qa["question"]) < 10:
            return False

        # Answer must be substantive (>2 tokens)
        if not qa["reference_answer"] or len(qa["reference_answer"].split()) < 2:
            return False

        # Question should not be too long
        if len(qa["question"].split()) > 50:
            return False

        return True

    @staticmethod
    def filter_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
        """Filter QA pairs to keep only valid ones."""
        return [qa for qa in qa_pairs if QAValidator.validate_qa_pair(qa)]
```

**Deliverables**:
- Question generation pipeline
- QA validation
- Parsing utilities

**Acceptance Criteria**:
- Can generate 10 questions per document
- Questions are parseable and valid
- Reference answers are substantive

---

### Phase 5: Evaluation Pipeline (Estimated: 1 day)

**Goals**:
- Answer questions from compressed representations
- Calculate metrics (EM, F1, semantic similarity)
- Compare to baselines

**Tasks**:

**5.1: Answerer**
```python
# src/evaluation/answerer.py

ANSWER_PROMPT = """
Answer the following question based ONLY on the provided context.
If the information needed to answer the question is not present in the context, respond with "UNKNOWN".

Context:
{context}

Question: {question}

Answer:
"""

class Answerer:
    def __init__(self, model):
        self.model = model

    def answer(self, context: str, question: str) -> str:
        """Answer question given context."""
        prompt = ANSWER_PROMPT.format(
            context=context,
            question=question
        )

        answer = self.model.generate(prompt, max_tokens=200)
        return answer.strip()
```

**5.2: Metrics Calculator**
```python
# src/evaluation/metrics.py
from typing import Dict
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

class MetricsCalculator:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_exact_match(self, prediction: str, reference: str) -> float:
        """Compute exact match score (0 or 1)."""
        pred_norm = self._normalize_text(prediction)
        ref_norm = self._normalize_text(reference)
        return float(pred_norm == ref_norm)

    def compute_f1(self, prediction: str, reference: str) -> float:
        """Compute F1 score (token overlap)."""
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if prediction.upper() == "UNKNOWN" or not prediction.strip():
            return 0.0

        emb1 = self.similarity_model.encode(prediction, convert_to_numpy=True)
        emb2 = self.similarity_model.encode(reference, convert_to_numpy=True)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def compute_all_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            "exact_match": self.compute_exact_match(prediction, reference),
            "f1": self.compute_f1(prediction, reference),
            "semantic_similarity": self.compute_semantic_similarity(prediction, reference),
            "is_unknown": float(prediction.upper().strip() == "UNKNOWN")
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import string
        # Lowercase, remove punctuation, extra spaces
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
```

**Deliverables**:
- Answering pipeline
- Metrics calculation (EM, F1, semantic similarity)
- Result storage

**Acceptance Criteria**:
- Can answer questions from both full and compressed contexts
- Metrics are calculated correctly
- Results match expected ranges

---

### Phase 6: Baselines Implementation (Estimated: 0.5 days)

**Goals**:
- Implement all baseline comparisons
- Ensure fair comparison

**Tasks**:

**6.1: Full-Context Baseline**
```python
# src/baselines/full_context.py
from ..evaluation.answerer import Answerer

class FullContextBaseline:
    def __init__(self, model):
        self.answerer = Answerer(model)

    def answer(self, document: str, question: str) -> str:
        """Answer question with full document context."""
        return self.answerer.answer(document, question)
```

**6.2: Human Summary Baseline**
```python
# src/baselines/human_summary.py
from ..compression.compressor import Compressor
from ..compression.prompts import HUMAN_SUMMARY_PROMPT

class HumanSummaryBaseline:
    def __init__(self, model):
        self.compressor = Compressor(model)

    def compress(self, document: str, token_limit: int = 1500):
        """Create human-readable summary."""
        return self.compressor.compress(
            document,
            HUMAN_SUMMARY_PROMPT,
            token_limit
        )
```

**6.3: Random Sampling Baseline**
```python
# src/baselines/random_sample.py
import random
from typing import Dict, Any

class RandomSampleBaseline:
    def __init__(self, token_counter, seed=42):
        self.token_counter = token_counter
        self.seed = seed

    def compress(self, document: str, token_limit: int = 1500) -> Dict[str, Any]:
        """Randomly sample tokens from document."""
        random.seed(self.seed)

        # Split into tokens (approximate)
        tokens = document.split()

        # Sample randomly
        if len(tokens) <= token_limit:
            sampled = tokens
        else:
            sampled = random.sample(tokens, token_limit)

        compressed = ' '.join(sampled)

        return {
            "compressed": compressed,
            "token_count": self.token_counter.count(compressed),
            "compliant": True,
            "original_tokens": self.token_counter.count(document),
            "compression_ratio": len(tokens) / len(sampled)
        }
```

**Deliverables**:
- All baseline implementations
- Consistent interface

**Acceptance Criteria**:
- All baselines produce comparable outputs
- Can run baselines on same documents as main experiment

---

### Phase 7: End-to-End Pilot (Estimated: 0.5 days)

**Goals**:
- Run complete pipeline on 10 documents
- Validate all components work together
- Identify issues

**Tasks**:

**7.1: Pilot Experiment Script**
```python
# experiments/pilot/run_pilot.py
# NOTE: Install package in development mode first: pip install -e .
# This avoids sys.path manipulation

from pathlib import Path
import pandas as pd

from src.models.openai_model import OpenAIModel
from src.data.loader import DatasetLoader
from src.compression.compressor import Compressor
from src.compression.prompts import SELF_COMPRESSION_PROMPT
from src.supervision.question_generator import QuestionGenerator
from src.evaluation.answerer import Answerer
from src.evaluation.metrics import MetricsCalculator

def run_pilot():
    # Setup - Use GPT-5.1 Instant (Nov 2025)
    # For cost savings, could use "gpt-4o" or wait for gpt-5-mini
    model = OpenAIModel(
        model_name="gpt-5.1-chat-latest",  # GPT-5.1 Instant
        temperature=0.0,
        seed=42,  # For reproducibility
        reasoning_effort="none"  # Disable adaptive reasoning for faster responses
    )

    # Load data
    documents = DatasetLoader.load_cnn_dailymail(num_samples=10)

    # Initialize components
    compressor = Compressor(model)
    qa_generator = QuestionGenerator(model)
    answerer = Answerer(model)
    metrics_calc = MetricsCalculator()

    results = []

    for doc in documents:
        print(f"\nProcessing document {doc['id']}...")

        # Compress
        compressed = compressor.compress(
            doc['text'],
            SELF_COMPRESSION_PROMPT,
            token_limit=1500
        )

        # Generate QA
        qa_pairs = qa_generator.generate_qa_pairs(doc['text'])

        # Answer questions
        for qa in qa_pairs:
            # Full context
            answer_full = answerer.answer(doc['text'], qa['question'])

            # Compressed context
            answer_comp = answerer.answer(compressed['compressed'], qa['question'])

            # Metrics
            metrics_full = metrics_calc.compute_all_metrics(
                answer_full, qa['reference_answer']
            )
            metrics_comp = metrics_calc.compute_all_metrics(
                answer_comp, qa['reference_answer']
            )

            results.append({
                "doc_id": doc['id'],
                "question": qa['question'],
                "reference": qa['reference_answer'],
                "answer_full": answer_full,
                "answer_compressed": answer_comp,
                "metrics_full": metrics_full,
                "metrics_compressed": metrics_comp,
                "compression_ratio": compressed['compression_ratio']
            })

    # Save results
    df = pd.DataFrame(results)

    # Create results directory if it doesn't exist
    results_dir = Path("results/pilot")
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / "pilot_results.csv", index=False)

    print("\nPilot complete!")
    print(f"Processed {len(documents)} documents, {len(results)} QA pairs")

    return df

if __name__ == "__main__":
    run_pilot()
```

**7.2: Manual Inspection**
- Review 10 compressed outputs
- Check if models use non-standard encodings
- Validate questions are answerable
- Check metric calculations

**Deliverables**:
- Working end-to-end pipeline
- Pilot results on 10 documents
- Issues list and fixes

**Acceptance Criteria**:
- Pipeline completes without errors
- Results are reasonable
- All components integrate correctly

---

### Phase 8: Main Experiment (Estimated: 1 day)

**Goals**:
- Run full experiment on 100 documents
- Test all compression variants and baselines
- Generate comprehensive results

**Tasks**:

**8.1: Experiment Configuration** (Updated for GPT-5.1)
```yaml
# experiments/configs/main_config.yaml
experiment:
  name: "main_self_compression"
  num_documents: 100
  dataset: "cnn_dailymail"
  seed: 42

model:
  # Use GPT-5.1 Instant (November 2025) - adaptive reasoning, fast
  name: "gpt-5.1-chat-latest"
  # Alternative models:
  # - "gpt-5.1-thinking" for deep reasoning (slower, more expensive)
  # - "gpt-5" for base GPT-5
  # - "gpt-4o" for cost savings / fallback
  temperature: 0.0
  seed: 42
  reasoning_effort: "none"  # Options: "none", "minimal", "medium", "high"

compression:
  token_limit: 1500
  variants:
    - name: "self_compression"
      prompt: "SELF_COMPRESSION_PROMPT"
    - name: "enhanced_self_compression"
      prompt: "ENHANCED_SELF_COMPRESSION_PROMPT"
    - name: "human_summary"
      prompt: "HUMAN_SUMMARY_PROMPT"

baselines:
  - "full_context"
  - "human_summary"
  - "random_sample"
  - "no_context"

evaluation:
  questions_per_document: 10
  metrics:
    - "exact_match"
    - "f1"
    - "semantic_similarity"
```

**8.2: Main Experiment Runner**
- Load configuration
- Iterate through all documents
- For each document:
  - Generate compressions (all variants)
  - Generate questions
  - Answer from all contexts (full, compressed variants, baselines)
  - Calculate metrics
  - Save intermediate results
- Aggregate results

**8.3: Progress Monitoring**
- Progress bar for documents
- Estimated time remaining
- Cost tracking (API calls)
- Error handling and recovery

**Deliverables**:
- Complete results on 100 documents
- Results for all compression variants and baselines
- Comprehensive metrics

**Acceptance Criteria**:
- All 100 documents processed successfully
- Results saved in structured format
- No data loss

---

### Phase 9: Analysis and Visualization (Estimated: 1 day)

**Goals**:
- Statistical analysis of results
- Generate visualizations
- Create comprehensive report

**Tasks**:

**9.1: Statistical Analysis**
```python
# src/analysis/statistics.py
from scipy import stats
import pandas as pd
import numpy as np

class StatisticalAnalyzer:
    @staticmethod
    def compare_conditions(df: pd.DataFrame, condition_a: str, condition_b: str):
        """Compare two experimental conditions."""
        # Paired t-test
        a_scores = df[df['condition'] == condition_a]['f1']
        b_scores = df[df['condition'] == condition_b]['f1']

        t_stat, p_value = stats.ttest_rel(a_scores, b_scores)

        # Effect size (Cohen's d)
        mean_diff = a_scores.mean() - b_scores.mean()
        pooled_std = np.sqrt((a_scores.std()**2 + b_scores.std()**2) / 2)
        cohens_d = mean_diff / pooled_std

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "mean_a": a_scores.mean(),
            "mean_b": b_scores.mean(),
            "std_a": a_scores.std(),
            "std_b": b_scores.std()
        }
```

**9.2: Visualization Suite**
- Bar charts: Performance by condition
- Box plots: Score distributions
- Scatter plots: Compression ratio vs. performance
- Heatmaps: Performance by question type
- Line plots: Performance degradation with compression

**9.3: Report Generator**
- Executive summary
- Methodology description
- Results tables
- Visualizations
- Discussion and conclusions
- Limitations and future work

**Deliverables**:
- Statistical analysis results
- 10+ visualizations
- Comprehensive markdown/HTML report
- LaTeX-ready tables

**Acceptance Criteria**:
- All statistical tests complete
- Visualizations are publication-quality
- Report is comprehensive and clear

---

## Success Criteria

### Technical Success
- ✅ Pipeline processes 100+ documents without errors
- ✅ All components integrate correctly
- ✅ Metrics are calculated accurately
- ✅ Results are reproducible (fixed seeds)

### Scientific Success
- ✅ Clear comparison between self-compression and baselines
- ✅ Statistical significance determined
- ✅ Compression strategies are analyzed qualitatively
- ✅ Answers to research questions:
  1. Does self-compression outperform human-readable summaries?
  2. What information is preserved vs. lost?
  3. Do models develop novel encoding schemes?
  4. How does performance vary by question type?

### Publication Readiness
- ✅ Comprehensive documentation
- ✅ Reproducible experiments
- ✅ Clear methodology
- ✅ Publication-quality figures
- ✅ Statistical rigor
- ✅ Discussion of limitations

---

## Reproducibility Checklist

- [ ] All random seeds fixed
- [ ] Model versions documented
- [ ] All prompts logged
- [ ] Full experiment configuration saved
- [ ] API calls logged with timestamps
- [ ] Raw outputs saved (not just metrics)
- [ ] Code is version controlled
- [ ] Dependencies pinned (requirements.txt)
- [ ] Environment variables documented
- [ ] README with setup instructions
- [ ] Tests pass
- [ ] Results can be regenerated from saved data

---

## Resource Estimates

### Computational Resources

**API-Based (GPT-5.1-chat-latest)**:
- Pilot (10 docs): ~200 API calls, **$TBD** (pricing not yet published for GPT-5.1), 1-2 hours
- Main (100 docs): ~2000 API calls, **$TBD** (estimate: $50-150 based on GPT-4 pricing), 8-12 hours
- Extended (1000 docs): ~20,000 API calls, **$TBD** (estimate: $500-1500), 2-3 days

**Cost Fallback Options**:
- Use `gpt-4o` for lower costs (~30-50% of GPT-5 pricing)
- Use `gpt-3.5-turbo` via Chat Completions API for budget prototyping (~$10-30 for main experiment)

**NOTE**: GPT-5.1 pricing not yet publicly available as of November 2025. Estimates based on historical GPT-4 → GPT-5 pricing patterns. Verify actual costs before running large experiments.

**Local Models (Llama-2-7B)**:
- Requires: GPU with 16GB VRAM
- Pilot: 2-3 hours
- Main: 1-2 days
- Extended: 1-2 weeks

### Development Time

- Phase 0 (Setup): 2-4 hours
- Phase 1 (Infrastructure): 1 day
- Phase 2 (Data): 0.5 days
- Phase 3 (Compression): 0.5 days
- Phase 4 (QA Generation): 0.5 days
- Phase 5 (Evaluation): 1 day
- Phase 6 (Baselines): 0.5 days
- Phase 7 (Pilot): 0.5 days
- Phase 8 (Main Experiment): 1 day
- Phase 9 (Analysis): 1 day

**Total: ~7 working days**

### Storage Requirements

- Raw datasets: ~500 MB
- Generated compressions: ~100 MB
- QA pairs: ~50 MB
- Results and logs: ~200 MB
- Figures: ~50 MB

**Total: ~1 GB**

---

## Risk Mitigation

### Technical Risks

**Risk**: API rate limits or downtime
**Mitigation**: Implement exponential backoff, save progress frequently, use multiple API keys

**Risk**: Token counting inaccuracies
**Mitigation**: Use official tokenizers (tiktoken), validate with manual checks

**Risk**: Data corruption
**Mitigation**: Save all raw outputs, version control, checksums

### Scientific Risks

**Risk**: Models don't use novel encodings
**Mitigation**: Try enhanced prompts, analyze what models naturally produce, still valuable result

**Risk**: High variance in LLM outputs
**Mitigation**: Use temperature=0, run multiple seeds, report confidence intervals

**Risk**: Question generation quality issues
**Mitigation**: Manual validation in pilot, automatic filters, use human-annotated QA as backup

---

## Next Steps After Implementation

1. **Write Paper**: Structure findings for publication (arXiv or conference)
2. **Share Code**: Release on GitHub with MIT license
3. **Create Demo**: Interactive web demo showing compression and QA
4. **Extend Research**:
   - Test with more models (GPT-4, Claude, Gemini)
   - Try different domains (code, scientific papers)
   - Multi-modal compression (images + text)
   - Fine-tuning for better compression

---

## Conclusion

This implementation plan provides a structured, phased approach to testing LLM self-compression for question answering. The plan is:

- ✅ **Tractable**: 7 days development + modest compute costs
- ✅ **Rigorous**: Statistical analysis, multiple baselines, comprehensive metrics
- ✅ **Reproducible**: Fixed seeds, logging, version control
- ✅ **Extensible**: Modular design allows easy additions
- ✅ **Novel**: Tests a unique research question not addressed in prior work

With this plan, we can systematically investigate whether LLMs can create effective self-compression schemes when freed from human-readability constraints.
