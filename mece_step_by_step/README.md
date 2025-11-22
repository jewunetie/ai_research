# MECE Step by Step Reasoning

> Testing whether MECE (Mutually Exclusive, Collectively Exhaustive) constraints improve reasoning completeness in LLMs

## Overview

This research project evaluates whether prompting language models to use the MECE principle improves their reasoning quality, particularly for problems requiring case analysis.

**Hardware**: M4 Max MacBook Pro (Apple Silicon optimized)
**Model**: Qwen3-0.6B-Instruct
**Domain**: Math case analysis problems
**Comparison**: MECE prompting vs Baseline CoT

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M4 Max recommended)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
cd mece_step_by_step

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Run Evaluation

```bash
# Phase 1: Create dataset
python scripts/create_dataset.py

# Phase 2-5: Run evaluation
python scripts/run_evaluation.py

# Phase 7: Analyze results
python scripts/analyze_results.py
```

## Project Structure

```
mece_step_by_step/
├── CLAUDE.md                   # Project description
├── RESEARCH.md                 # Literature review
├── IMPLEMENTATION.md           # Detailed implementation plan
├── REVIEW_IMPLEMENTATION.md    # Bug fixes and improvements
├── config.yaml                 # Configuration
├── pyproject.toml              # Dependencies (uv/pip)
├── README.md                   # This file
│
├── data/
│   ├── math_case_analysis.json      # 50 math problems
│   └── results/                      # Evaluation outputs
│       ├── baseline_results.json
│       └── mece_results.json
│
├── src/
│   ├── models/
│   │   └── qwen_inference.py        # MLX-optimized Qwen3-0.6B
│   ├── prompts/
│   │   ├── baseline_prompt.py       # Standard CoT
│   │   └── mece_prompt.py           # MECE prompting
│   ├── metrics/
│   │   ├── mutual_exclusivity.py    # ME score calculation
│   │   ├── collective_exhaustiveness.py  # CE score calculation
│   │   └── accuracy.py              # Answer accuracy
│   ├── evaluation/
│   │   ├── evaluator.py             # Main evaluation harness
│   │   └── parser.py                # Parse reasoning steps
│   └── utils/
│       └── config.py                # Config loading
│
├── scripts/
│   ├── create_dataset.py            # Generate 50 problems
│   ├── run_evaluation.py            # Main evaluation script
│   └── analyze_results.py           # Results analysis
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Interactive analysis
│
└── tests/
    ├── test_metrics.py
    └── test_parsers.py
```

## Research Questions

1. **Does MECE prompting improve mutual exclusivity?** (ME Score)
2. **Does MECE prompting improve collective exhaustiveness?** (CE Score)
3. **Does MECE prompting maintain or improve accuracy?**
4. **What patterns emerge in when MECE helps vs doesn't help?**

## Metrics

### Mutual Exclusivity (ME) Score
- Embedding-based similarity between reasoning steps
- Case condition overlap detection
- **Higher is better** (0-1 scale)

### Collective Exhaustiveness (CE) Score
- Case enumeration: covered cases / required cases
- Condition coverage verification
- **Higher is better** (0-1 scale)

### Combined MECE Score
- Harmonic mean of ME and CE scores

### Accuracy
- Exact match with ground truth solutions

## Dataset

50 math case analysis problems across 4 categories:

1. **Absolute Value Equations** (15 problems)
   - Example: Solve |x - 3| = 5
   - Required cases: x - 3 ≥ 0, x - 3 < 0

2. **Piecewise Functions** (15 problems)
   - Example: Evaluate f(x) = {x² if x < 0, 2x if x ≥ 0}
   - Required cases: Different ranges

3. **Sign Analysis** (10 problems)
   - Example: For what values is (x-1)(x+2) > 0?
   - Required cases: Different sign regions

4. **Range-Based Problems** (10 problems)
   - Example: Find integer solutions to x² < 10
   - Required cases: Positive, negative, zero

## Configuration

Edit `config.yaml` to customize:
- Model parameters (temperature, max_tokens)
- Thinking mode settings
- Evaluation parameters
- Output directories

## Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/
```

## Hardware Optimization (M4 Max)

This project uses MLX (Apple's ML framework) for optimal performance on Apple Silicon:

- **MLX**: Native Apple Silicon acceleration
- **MPS**: Metal Performance Shaders for GPU operations
- **Unified Memory**: Efficient memory usage on M4 Max

Expected inference speed: **~5-10 seconds per problem**

## Results

Results will be saved to `data/results/` including:
- Full model responses
- Parsed reasoning steps
- ME/CE/accuracy scores
- Statistical analysis
- Visualizations

## Timeline

- **Day 1**: Dataset creation + model setup
- **Day 2**: Prompt engineering + metrics
- **Day 3**: Evaluation pipeline + full evaluation
- **Day 4-5**: Analysis + visualization
- **Total**: ~5-7 days

## Success Criteria

### Minimum
- ✅ Pipeline runs end-to-end
- ✅ Metrics computed for all 50 problems
- ✅ Measurable differences in ME/CE scores

### Strong
- ✅ MECE improves ME score by >10%
- ✅ MECE improves CE score by >10%
- ✅ Accuracy maintained or improved
- ✅ Statistically significant (p < 0.05)

### Outstanding
- ✅ All strong criteria met
- ✅ Accuracy improvement >5%
- ✅ Clear patterns identified
- ✅ Generalizable insights

## Citations

See `RESEARCH.md` for comprehensive literature review including:
- Tree-of-Thoughts (Yao et al., 2023)
- Self-Consistency (Wang et al., 2022)
- Qwen3 models (Alibaba, 2025)
- And 30+ related papers

## License

MIT

## Contributing

This is a research project. For questions or suggestions, please open an issue.

---

**Status**: Phase 0 Complete (Planning & Documentation)
**Next**: Phase 1 (Dataset Creation)
