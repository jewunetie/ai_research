# Forward-Forward with Backprop

Research implementation exploring hybrid training schemes that combine Hinton's Forward-Forward algorithm with standard backpropagation.

## Quick Start

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Running Experiments

```bash
# Run baseline backpropagation on MNIST
python experiments/run_experiment.py --config experiments/configs/mnist_baseline_bp.yaml

# Run sequential phased approach
python experiments/run_experiment.py --config experiments/configs/mnist_sequential_phased.yaml

# Run detached interface approach
python experiments/run_experiment.py --config experiments/configs/mnist_detached_interface.yaml

# Run all baselines
python experiments/run_baselines.py --dataset mnist
```

### Analyzing Results

```bash
# Compare all results
python experiments/compare_results.py

# Or use Jupyter notebooks
jupyter notebook notebooks/03_analyze_results.ipynb
```

## Project Structure

```
forward_forward_backprop/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── training/          # Training loops
│   ├── data/              # Data loaders
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Utilities
├── experiments/           # Experiment scripts and configs
├── results/               # Experiment outputs
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project overview and concept
- [RESEARCH.md](RESEARCH.md) - Literature review
- [INTERMIXED_ARCHITECTURES_RESEARCH.md](INTERMIXED_ARCHITECTURES_RESEARCH.md) - Deep dive on hybrid architectures
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Detailed implementation plan

## Approaches Implemented

1. **Sequential Phased**: FF pretraining → BP fine-tuning
2. **Detached Interface**: Simultaneous FF+BP with gradient detachment
3. **Block-wise Hybrid**: BP within blocks, FF between blocks

## Baselines

1. Pure Backpropagation
2. Pure Forward-Forward
3. Random Initialization + BP Classifier
4. Autoencoder Pretraining + BP Fine-tuning

## Hardware Support

- ✅ Apple Silicon (MPS backend)
- ✅ NVIDIA GPUs (CUDA)
- ✅ CPU

## Results

Results will be saved to `results/` directory with:
- Training logs and metrics
- Model checkpoints
- Visualization plots
- Comparison tables

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{forward_forward_backprop2025,
  title={Forward-Forward with Backprop: Exploring Hybrid Training Schemes},
  author={Research Project},
  year={2025},
  url={https://github.com/yourusername/forward_forward_backprop}
}
```

## References

1. Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345
2. Scalable Forward-Forward Algorithm (2025). arXiv:2501.03176
3. See RESEARCH.md for complete list of references
