# LLM Self-Compression for Downstream Question Answering

## Overview

This project explores a novel research question: **Can large language models create and use their own semantic compression schemes for downstream question answering?**

## Core Idea

The experimental protocol consists of three distinct roles:

1. **Compressor LLM**: Given an original text document, the compressor LLM produces a highly compressed representation (possibly non-human-readable) that it can still interpret later.

2. **Supervisor LLM**: Generates questions from the original text along with reference answers, which will be used to evaluate the compression quality.

3. **Answerer LLM**: Must answer the supervisor's questions using **only** the compressed representation, without access to the original text.

The key insight is to test how well LLMs can create and use their own semantic compression schemes when the compression doesn't need to be human-readable, only machine-interpretable by the same model.

## The Compression Prompt

The specific compression prompt we are studying:

> "Now summarize everything in this text in as much detail as possible, but compress it as much as possible into a format that you can still read. It does not need to be human readable. You do not need to use a common character set, all that matters is we can pick back up right where we left off if I were to start a new conversation with you. You are limited to 1500 tokens."

### Key Characteristics of This Prompt

- **Self-directed compression**: The model compresses for itself, not for humans
- **Non-human-readable**: Allows the model to use arbitrary encodings, symbols, or compression schemes
- **Fixed token budget**: 1500 tokens maximum
- **Preservation goal**: Maximum detail retention within the constraint
- **Novel character sets allowed**: The model can invent its own notation

## Research Questions

1. **Effectiveness**: How well can LLMs answer questions from self-generated compressed representations compared to the original text?

2. **Compression Strategies**: What compression strategies do LLMs naturally develop when unconstrained by human readability?

3. **Information Retention**: What types of information are preserved vs. lost in self-compression?

4. **Generalization**: Does self-compression work across different document types and question types?

5. **Model Dependency**: Do compressed representations transfer across different models, or are they model-specific?

## Hypotheses

- **H1**: LLMs can develop effective compression schemes that preserve semantic information better than random sampling or simple truncation.

- **H2**: Self-compressed representations (non-human-readable) will outperform human-readable summaries of the same length for downstream QA tasks.

- **H3**: Factual/detail-oriented questions will suffer more from compression than inferential/conceptual questions.

- **H4**: Compression quality will vary by document structure (narrative vs. expository vs. technical).

## Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: uv (for fast, reliable dependency management)
- **ML Framework**: PyTorch (if needed for model fine-tuning or custom implementations)
- **Model Access**: Hugging Face Transformers (for loading and running LLMs)
- **Datasets**: Hugging Face Datasets (for document collections and QA benchmarks)
- **Evaluation**: Standard QA metrics (EM, F1, semantic similarity)

## Experimental Design (High-Level)

### Pipeline

```
Original Document (D)
    ↓
    ├──→ [Compressor LLM] ──→ Compressed Representation (C, ≤1500 tokens)
    │                               ↓
    │                               └──→ [Answerer LLM] ──→ Predicted Answers (Â₁...Âₙ)
    │                                           ↑
    └──→ [Supervisor LLM] ──→ Questions (Q₁...Qₙ) + Reference Answers (A₁...Aₙ)
```

### Evaluation

Compare predicted answers (Â) to reference answers (A) using:
- Exact Match (EM)
- F1 Score
- Semantic Similarity (e.g., BERTScore, embedding cosine similarity)
- Human evaluation (for qualitative analysis)

## Success Criteria

1. **Functional**: System successfully compresses documents, generates questions, and evaluates answers
2. **Comparative**: Self-compression shows measurable performance vs. baselines
3. **Analytical**: Clear analysis of what information is retained/lost
4. **Reproducible**: All experiments can be replicated with documented prompts, seeds, and model versions

## Directory Structure

```
llm_self_compression_qa/
├── CLAUDE.md                   # This file
├── RESEARCH.md                 # Literature review and related work
├── IMPLEMENTATION.md           # Detailed implementation plan
├── requirements.txt            # Python dependencies
├── pyproject.toml             # uv project configuration
├── src/
│   ├── compression/           # Compression pipeline
│   ├── supervision/           # Question generation
│   ├── evaluation/            # QA evaluation
│   ├── baselines/             # Baseline implementations
│   ├── models/                # Model loading and management
│   └── utils/                 # Utilities and helpers
├── data/                      # Input documents and generated datasets
├── results/                   # Experimental results
├── notebooks/                 # Analysis notebooks
└── tests/                     # Unit tests
```

## Next Steps

1. Literature review on semantic compression and prompt compression methods
2. Identify relevant datasets and documents for testing
3. Implement the core pipeline
4. Run baseline experiments
5. Analyze results and iterate
