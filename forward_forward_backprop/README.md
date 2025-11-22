# Forward-Forward with Backprop: Research Implementation

Research prototype exploring hybrid training schemes combining Hinton's Forward-Forward (FF) algorithm with traditional Backpropagation (BP).

## Overview

This project investigates novel architectures that intermix FF and BP training:

1. **Sequential Phased**: FF pretraining (unsupervised) → BP classifier (frozen) → Fine-tune all
2. **Detached Interface**: Simultaneous FF+BP training with gradient detachment
3. **Baselines**: Pure BP and Pure FF for comparison

**Key Research Questions:**
- Can FF pretraining improve BP fine-tuning?
- How do hybrid approaches compare to pure BP/FF?
- What representations do FF layers learn?

## Quick Start

### Installation

```bash
# Clone repository
cd forward_forward_backprop

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
uv pip install -e .
```

### Run Your First Experiment

```bash
# Run MNIST baseline (pure backprop)
python experiments/run_experiment.py --config experiments/configs/mnist_baseline_bp.yaml

# Run sequential phased hybrid
python experiments/run_experiment.py --config experiments/configs/mnist_sequential_phased.yaml

# Run detached interface approach
python experiments/run_experiment.py --config experiments/configs/mnist_detached_interface.yaml

# Run pure FF baseline
python experiments/run_experiment.py --config experiments/configs/mnist_baseline_ff.yaml
```

### Compare Results

```bash
# Compare all MNIST experiments
python experiments/compare_results.py results/mnist_*

# Detailed view
python experiments/compare_results.py results/mnist_* --detailed
```

## Project Structure

```
forward_forward_backprop/
├── src/
│   ├── models/
│   │   ├── ff_layer.py          # Forward-Forward layers and network
│   │   └── mlp.py               # MLP and HybridFFBPModel
│   ├── training/
│   │   ├── ff_trainer.py        # Pure FF trainer
│   │   ├── bp_trainer.py        # Pure BP trainer
│   │   ├── sequential_phased.py # Sequential phased hybrid
│   │   └── detached_interface.py # Detached interface hybrid
│   ├── data/
│   │   ├── datasets.py          # Dataset loaders (MNIST, Fashion-MNIST, CIFAR-10)
│   │   └── augmentation.py      # Negative sample generation, label embedding
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, linear probing, goodness analysis
│   │   └── visualization.py     # t-SNE, training curves, confusion matrices
│   └── utils/
│       ├── config.py            # YAML config loading
│       └── device.py            # Device detection (MPS/CUDA/CPU)
├── experiments/
│   ├── configs/                 # Experiment configurations
│   │   ├── mnist_baseline_bp.yaml
│   │   ├── mnist_baseline_ff.yaml
│   │   ├── mnist_sequential_phased.yaml
│   │   ├── mnist_detached_interface.yaml
│   │   ├── fashion_mnist_sequential_phased.yaml
│   │   └── cifar10_sequential_phased.yaml
│   ├── run_experiment.py        # Main experiment runner
│   └── compare_results.py       # Results comparison tool
├── tests/
│   └── test_bug_fixes.py        # Validation test suite
└── results/                      # Experiment outputs (created at runtime)
```

## Experiment Configurations

### Available Experiments

| Config | Approach | Dataset | Description |
|--------|----------|---------|-------------|
| `mnist_baseline_bp.yaml` | Pure BP | MNIST | Standard backprop baseline |
| `mnist_baseline_ff.yaml` | Pure FF | MNIST | Forward-Forward baseline |
| `mnist_sequential_phased.yaml` | Hybrid | MNIST | **Main approach**: 3-phase training |
| `mnist_detached_interface.yaml` | Hybrid | MNIST | **Novel**: Simultaneous FF+BP |
| `fashion_mnist_sequential_phased.yaml` | Hybrid | Fashion-MNIST | Scalability test |
| `cifar10_sequential_phased.yaml` | Hybrid | CIFAR-10 | Color images test |

### Sequential Phased Training (3 Phases)

**Phase 1: FF Pretraining (Unsupervised)**
- Train FF layers with threshold loss
- Learn robust representations without labels
- 100 epochs, lr=0.03

**Phase 2a: BP Classifier (Frozen FF)**
- Freeze FF layers, train classifier only
- Supervised learning on frozen features
- 50 epochs, lr=0.001

**Phase 2b: Fine-tune All (Optional)**
- Unfreeze everything, fine-tune end-to-end
- Polish representations with BP
- 50 epochs, lr=0.0001

### Detached Interface Training

- **Simultaneous FF+BP**: Both run in parallel
- **Gradient Detachment**: BP doesn't affect FF layers
- **Two Optimizers**: Separate for FF and BP components
- **Novel Approach**: Not found in existing literature

## Running Experiments

### Basic Usage

```bash
python experiments/run_experiment.py --config experiments/configs/<config_file>.yaml
```

### What Happens During Training

1. **Config Loading**: Reads experiment parameters from YAML
2. **Device Setup**: Auto-detects MPS (Apple Silicon) / CUDA / CPU
3. **Data Loading**: Downloads and prepares dataset with normalization
4. **Model Creation**: Builds architecture based on config
5. **Training**: Runs training with progress logging
6. **Checkpointing**: Saves model at specified intervals
7. **Evaluation**: Computes final test accuracy
8. **Results**: Saves outputs to `results/<experiment_name>/`

## Validation Tests

Run validation tests to verify all components work:

```bash
python tests/test_bug_fixes.py
```

This tests:
- All imports
- Device detection
- Data loading with normalization
- Negative sample generation (all strategies)
- Label embedding
- Model creation
- Forward/backward passes
- Trainer validation

## Results and Analysis

### Results Directory Structure

After running experiments, `results/` contains:

```
results/
├── mnist_baseline_bp/
│   ├── config_*.yaml          # Experiment config
│   ├── results_*.txt          # Text summary
│   ├── checkpoint_*.pt        # Model checkpoints
│   └── final_model.pt         # Final trained model
├── mnist_sequential_phased/
│   └── ...
└── ...
```

### Comparing Results

```bash
# Compare all MNIST experiments
python experiments/compare_results.py results/mnist_*
```

Output shows:
- Sorted table by final accuracy
- Summary statistics (best/worst/average/std)
- Grouped by approach and dataset

## Configuration Guide

### Negative Sample Strategies

- **random_label**: Replace label with random class (default)
- **augmented**: Add Gaussian noise to images
- **shuffled**: Shuffle pixels within each image
- **batch_shuffled**: Shuffle images within batch

## Research Background

### Forward-Forward Algorithm (Hinton 2022)

- **Local Learning**: Each layer learns independently
- **Two Passes**: One with positive data (real), one with negative (corrupted)
- **Goodness Function**: Sum of squared activations
- **Threshold Loss**: Push positive above threshold, negative below

### Our Contributions

1. **Sequential Phased Training**: Novel phased approach recommended by SCFF paper but not implemented
2. **Detached Interface**: Original architecture for simultaneous FF+BP
3. **Comprehensive Comparison**: Systematic evaluation across baselines
4. **Representation Analysis**: Linear probing and visualization tools

### Key Papers

- Hinton (2022): "The Forward-Forward Algorithm" - Original FF
- Khoi et al. (2024): "SCFF" - Supervised Contrastive FF achieving 98.70% on MNIST
- Sun et al. (2025): "SFF" - Sparse FF with block-wise hybrid (81.38% on CIFAR-10)
- Song et al. (2025): "Deep-CBN" - Modular stage-wise hybrid for molecular prediction

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project overview and concept
- [RESEARCH.md](RESEARCH.md) - Comprehensive literature review
- [INTERMIXED_ARCHITECTURES_RESEARCH.md](INTERMIXED_ARCHITECTURES_RESEARCH.md) - Deep dive on hybrid architectures
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Detailed implementation plan

## Hardware Requirements

**Minimum:**
- CPU: Any modern CPU
- RAM: 8GB
- Storage: 2GB

**Recommended:**
- GPU: Apple Silicon (MPS) or NVIDIA (CUDA)
- RAM: 16GB
- Storage: 10GB (for multiple experiments)

**Training Times (MNIST, 200 total epochs):**
- M4 Max (MPS): ~10-15 minutes
- NVIDIA RTX 4090: ~5-8 minutes
- CPU only: ~30-45 minutes

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
uv pip install -e .
```

### Device Issues

```bash
# Force CPU - Edit config: device: "cpu"

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

- Reduce batch size in config
- Use fewer workers: `num_workers: 0`
- Reduce model size: smaller hidden dims

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ff_bp_hybrid_2025,
  title={Forward-Forward with Backprop: Hybrid Training Schemes},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/forward_forward_backprop}
}
```

## Acknowledgments

- Geoffrey Hinton for the Forward-Forward algorithm
- PyTorch team for the excellent framework
- Research papers that inspired this work (see RESEARCH.md)

---

**Status**: MVP Complete ✓
**Next Steps**: Run experiments, analyze results, extend to CIFAR-10/100
