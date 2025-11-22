# LLM Self-Compression for Question Answering

Research prototype investigating whether LLMs can create their own non-human-readable compression schemes for downstream question answering tasks.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Verify API Setup

**IMPORTANT**: Run this first to verify which OpenAI API version you have access to:

```bash
python verify_api.py
```

This will test whether the Responses API (March 2025) is available or if you need to use Chat Completions API.

### 4. Run Pilot Experiment

```bash
python scripts/pilot.py
```

## Project Structure

```
llm_self_compression_qa/
├── src/
│   ├── models/           # LLM interfaces
│   ├── compression/      # Compression pipeline
│   ├── data/            # Data loading and processing
│   ├── evaluation/      # Question generation and answering
│   └── baselines/       # Baseline implementations
├── scripts/             # Experiment scripts
├── results/             # Experimental results
└── docs/               # Documentation
```

## Documentation

- `CLAUDE.md` - Project overview and research specification
- `RESEARCH.md` - Literature review and related work
- `IMPLEMENTATION.md` - Detailed implementation plan and code examples

## API Compatibility

This project supports both:
- **Responses API** (March 2025+): `gpt-5.1-chat-latest`, `gpt-5.1-thinking`
- **Chat Completions API** (fallback): `gpt-4o`, `gpt-3.5-turbo`

The code automatically detects which API is available and uses appropriate fallbacks.

## Research Questions

1. Can LLMs create effective non-human-readable compression schemes?
2. How much information retention is possible at 1500 token limit?
3. Do self-compressed representations preserve QA-relevant information?
4. How does self-compression compare to human-readable summaries?

See `CLAUDE.md` for full research specification.
