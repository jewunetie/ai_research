# UNKNOWN Token Sink: Learning to Abstain on Out-of-Distribution Inputs

Research prototype for training language models with an explicit abstention mechanism using a special `<UNKNOWN>` token.

## Overview

This project explores whether language models can be trained to abstain from making predictions on out-of-distribution or semantically meaningless inputs by learning to output a special `<UNKNOWN>` token, rather than hallucinating plausible-sounding but incorrect responses.

### Key Innovation

- **Dedicated Abstention Token**: Add `<UNKNOWN>` to model vocabulary
- **Synthetic Gibberish Training**: Train on 4 types of nonsense data
  - Type 1: Repetitive tokens ("apple apple apple")
  - Type 2: Random sequences (incoherent token combinations)
  - Type 3: Semantic nulls ("Colorless green ideas sleep furiously")
  - Type 4: Corrupted real data (10%-70% token corruption)
- **Pattern-Based Abstention**: Model learns to recognize gibberish patterns, not just confidence thresholding

## Project Structure

```
unknown_token_sink/
├── src/
│   ├── data/              # Data generation and loading
│   ├── models/            # Model loading and modification
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation scripts
│   ├── baselines/         # Baseline implementations
│   └── utils/             # Utility functions
├── configs/               # YAML configuration files
├── scripts/               # Shell scripts for automation
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Generated datasets
├── output/                # Model checkpoints
├── logs/                  # TensorBoard logs
├── results/               # Evaluation results
└── docs/                  # Documentation
    ├── CLAUDE.md          # Project overview
    ├── RESEARCH.md        # Literature review
    ├── MODEL_SELECTION.md # Model and dataset choices
    ├── GIBBERISH_GENERATION.md  # Data generation specs
    ├── IMPLEMENTATION_DESIGN.md # Technical architecture
    └── IMPLEMENTATION.md  # Execution roadmap
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with 16GB+ VRAM (recommended: 24GB+)
- 32GB system RAM
- 50GB free disk space

### Installation

```bash
# Clone or navigate to project directory
cd /home/user/ai_research/unknown_token_sink

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run setup validation
python scripts/validate_setup.py
```

## Usage

### 1. Generate Training Data

```bash
# Generate all gibberish types and sample FineWeb-Edu
bash scripts/generate_all_data.sh
```

This creates:
- `data/train/`: 90K real + 8K gibberish examples
- `data/validation/`: 10K real + 1K gibberish examples
- `data/test/`: 5K real + 1K gibberish examples

### 2. Train Model

```bash
# Train with default configuration (10% gibberish ratio)
python src/training/train.py \
    --config configs/training_config.yaml \
    --model_config configs/model_config.yaml

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### 3. Evaluate

```bash
# Run full evaluation suite
bash scripts/run_evaluation.sh

# Or run individual evaluations
python src/evaluation/evaluate_indist.py          # In-distribution
python src/evaluation/evaluate_gibberish.py       # Synthetic gibberish
python src/evaluation/evaluate_squad.py           # SQuAD 2.0
python src/evaluation/evaluate_domain_shift.py    # PubMedQA
python src/evaluation/evaluate_hallucination.py   # TruthfulQA
```

### 4. Analyze Results

```bash
# Launch Jupyter notebook for analysis
jupyter notebook notebooks/02_analyze_results.ipynb
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

- **Base model**: Gemma 3 270M (primary) or SmolLM-360M (backup)
- **Max sequence length**: 512 tokens
- **FP16 training**: Enabled for memory efficiency

### Training Configuration (`configs/training_config.yaml`)

- **Epochs**: 3
- **Batch size**: 32 (effective, via gradient accumulation)
- **Learning rate**: 5e-5 with cosine schedule
- **Gibberish ratio**: 10% (tune to 15% if under-abstention)

### Evaluation Configuration (`configs/eval_config.yaml`)

- **Success criteria**:
  - In-dist UNKNOWN rate: <5%
  - Gibberish detection: >90%
  - SQuAD 2.0 unanswerable: >70% UNKNOWN
  - SQuAD 2.0 answerable: <10% UNKNOWN

## Success Criteria

### Minimum Viable Success ✅
- ✓ Training completes without errors
- ✓ In-dist perplexity within 10% of baseline
- ✓ In-dist UNKNOWN rate <5%
- ✓ Synthetic gibberish detection >90%

### Full Success 🎯
All above, plus:
- ✓ SQuAD 2.0 unanswerable detection >70%
- ✓ SQuAD 2.0 answerable UNKNOWN <10%
- ✓ TruthfulQA hallucination reduction >5%
- ✓ Outperforms baselines on coverage-accuracy

## Troubleshooting

### GPU Out of Memory
```yaml
# In configs/training_config.yaml, reduce batch size:
per_device_train_batch_size: 8  # Down from 16
gradient_accumulation_steps: 4  # Up from 2
```

### Training Diverges
```yaml
# Reduce learning rate and increase warmup:
learning_rate: 2.0e-5  # Down from 5e-5
warmup_ratio: 0.2      # Up from 0.1
```

### Over-Abstention (In-dist UNKNOWN >5%)
```yaml
# Reduce gibberish ratio:
gibberish_ratio: 0.05  # Down from 0.10

# Or reduce gibberish loss weight:
loss:
  gibberish_weight: 0.3  # Down from 1.0
```

### Under-Abstention (Gibberish detection <90%)
```yaml
# Increase gibberish ratio:
gibberish_ratio: 0.15  # Up from 0.10

# Or train longer:
num_train_epochs: 5  # Up from 3
```

## Development

### Running Tests
```bash
# Test data pipeline
python scripts/test_data_pipeline.py

# Test model loading
python scripts/test_token_integration.py
```

### Code Style
```bash
# Format code (if you have black installed)
black src/ scripts/

# Type checking (if you have mypy installed)
mypy src/
```

## Documentation

Comprehensive documentation available in the project:

- **[CLAUDE.md](CLAUDE.md)**: Project overview and core idea
- **[RESEARCH.md](RESEARCH.md)**: Comprehensive literature review (12 sections)
- **[MODEL_SELECTION.md](MODEL_SELECTION.md)**: Model and dataset research
- **[GIBBERISH_GENERATION.md](GIBBERISH_GENERATION.md)**: Complete data generation specs
- **[IMPLEMENTATION_DESIGN.md](IMPLEMENTATION_DESIGN.md)**: Technical architecture
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: Phased execution roadmap

## Citation

If you use this code or build upon this research, please cite:

```bibtex
@misc{unknown_token_sink_2025,
  title={In-Training-Distribution Knowledge Classification via UNKNOWN Token Sink},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/unknown-token-sink}
}
```

## License

MIT License (or specify your license)

## Acknowledgments

- **Base Model**: Google Gemma 3 270M
- **Dataset**: Hugging Face FineWeb-Edu
- **Evaluation**: SQuAD 2.0, PubMedQA, TruthfulQA

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Status**: Phase 0 (Project Setup) Complete ✅
**Next**: Phase 1 (Data Generation)
