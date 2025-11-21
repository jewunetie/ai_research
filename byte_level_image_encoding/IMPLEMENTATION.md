# Implementation Plan

## Overview

This document provides a **phased, incremental implementation plan** for the byte-level image encoding research project. Each phase builds on the previous one and can be executed independently.

**Timeline Estimate**: 2-3 weeks for Phases 1-3, +1 week for Phase 4 (optional)

---

## Phase 0: Project Setup (Day 1)

### Goals
- Set up Python environment with uv
- Install dependencies
- Create project structure
- Verify CIFAR-10 dataset loads

### Tasks

#### 0.1 Initialize uv Project
```bash
cd byte_level_image_encoding
uv init
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121
uv add pillow numpy matplotlib tqdm pyyaml
uv add pytest --dev
```

#### 0.2 Create Directory Structure
```bash
mkdir -p src/{data,models,training,utils}
mkdir -p configs scripts notebooks tests
touch src/{__init__,data/__init__,models/__init__,training/__init__,utils/__init__}.py
```

#### 0.3 Test CIFAR-10 Loading
Create `scripts/test_setup.py`:
```python
import torch
import torchvision

# Test CIFAR-10 download
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print(f"Dataset loaded: {len(dataset)} images")
print(f"Classes: {dataset.classes}")

# Test basic operations
img, label = dataset[0]
print(f"Image type: {type(img)}, size: {img.size}")
print(f"Label: {label} ({dataset.classes[label]})")
```

#### 0.4 Verify GPU Access
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Success Criteria
- ✅ uv environment created
- ✅ All dependencies installed
- ✅ CIFAR-10 downloads successfully
- ✅ GPU detected (if available)

---

## Phase 1: Baseline Implementation (Days 2-4)

### Goals
- Implement pixel-level baseline (ResNet-18)
- Train and validate on CIFAR-10
- Establish performance benchmark

### Tasks

#### 1.1 Implement Pixel Dataset (`src/data/cifar10_pixels.py`)
```python
class PixelLevelCIFAR10:
    def __init__(self, root, train=True, download=True):
        # Standard torchvision implementation with transforms
        pass

    @staticmethod
    def get_default_transforms(train=True):
        # RandomCrop, RandomHorizontalFlip, Normalize
        pass
```

**Test**: Load 10 samples, visualize

#### 1.2 Implement ResNet-18 Baseline (`src/models/resnet.py`)
```python
class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        # Adapt torchvision ResNet-18 for 32×32 images
        pass
```

**Test**: Forward pass with dummy input, verify output shape

#### 1.3 Implement Training Loop (`src/training/trainer.py`)
```python
class Trainer:
    def train_epoch(self): pass
    def validate(self): pass
    def train(self, num_epochs): pass
    def save_checkpoint(self, path): pass
```

**Test**: Train for 1 epoch, verify loss decreases

#### 1.4 Create Training Script (`scripts/train_resnet18.py`)
```python
def main():
    # Load config
    # Create data loaders
    # Initialize model, optimizer, scheduler
    # Create trainer
    # Train for num_epochs
    # Save results
```

#### 1.5 Create Configuration (`configs/resnet18_pixels.yaml`)
```yaml
experiment_name: "resnet18_cifar10_baseline"
model:
  name: "resnet18"
  num_classes: 10
training:
  num_epochs: 100
  batch_size: 128
  learning_rate: 0.1
  optimizer: "sgd"
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: "multistep"
  milestones: [50, 75]
```

#### 1.6 Train ResNet-18 Baseline
```bash
uv run python scripts/train_resnet18.py --config configs/resnet18_pixels.yaml
```

**Expected Results**:
- Training time: ~3 hours (100 epochs, A100)
- Final accuracy: 92-95% on test set
- Loss curves: Smooth convergence

#### 1.7 Analyze and Document Results
- Plot training/validation curves
- Measure inference time
- Count FLOPs (use `fvcore` or manual calculation)
- Save to `results/resnet18_baseline.json`

### Success Criteria
- ✅ ResNet-18 trains successfully
- ✅ Achieves ≥92% test accuracy
- ✅ Results documented (accuracy, time, FLOPs)
- ✅ Checkpoints saved

### Deliverables
- `src/data/cifar10_pixels.py`
- `src/models/resnet.py`
- `src/training/trainer.py`
- `scripts/train_resnet18.py`
- `configs/resnet18_pixels.yaml`
- `results/resnet18_baseline.json`
- `checkpoints/resnet18_best.pt`

---

## Phase 2: Byte-Level Implementation (Days 5-10)

### Goals
- Implement byte-level data loading
- Implement ByteFormer architecture
- Train on CIFAR-10 with byte sequences

### Tasks

#### 2.1 Implement Format Conversion Utils (`src/data/transforms.py`)
```python
def image_to_bytes(image: PIL.Image, format: str, quality: int = 75) -> List[int]:
    """Convert PIL Image to byte sequence."""
    pass

def pad_or_truncate(byte_seq: List[int], length: int) -> List[int]:
    """Pad with zeros or truncate to fixed length."""
    pass
```

**Test**: Convert CIFAR-10 samples to JPEG/PNG, verify byte sequences

#### 2.2 Analyze File Sizes
Create `scripts/analyze_byte_lengths.py`:
```python
# Convert 1000 CIFAR-10 images to JPEG/PNG
# Measure byte sequence lengths
# Plot distribution
# Determine appropriate max_length
```

**Output**: Histogram of byte lengths, recommended max_length

#### 2.3 Implement Byte Dataset (`src/data/cifar10_bytes.py`)
```python
class ByteLevelCIFAR10:
    def __init__(self, root, train=True, format='mixed',
                 jpeg_quality=75, max_length=8192):
        pass

    def __getitem__(self, idx):
        # Load image
        # Convert to bytes (JPEG or PNG based on format)
        # Pad/truncate
        # Return tensor
        pass
```

**Test**: Load batch, verify shapes, check format distribution

#### 2.4 Implement Shifted Window Attention (`src/models/layers.py`)
```python
class ShiftedWindowTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, window_size, shift=False):
        pass

    def forward(self, x):
        # Shift if needed
        # Window partition
        # Apply attention within windows
        # Reverse window partition
        # FFN
        pass
```

**Test**: Forward pass with dummy input, verify output shape

#### 2.5 Implement ByteFormer (`src/models/byteformer.py`)
```python
class ByteFormerCIFAR(nn.Module):
    def __init__(self, max_length=8192, d_model=192, nhead=3,
                 num_layers=6, window_size=128, downsample_layers=[2, 4]):
        # Byte embedding
        # Positional encoding
        # Initial downsampling (strided conv)
        # Transformer blocks
        # Hierarchical downsampling
        # Classification head
        pass

    def forward(self, byte_seq):
        pass
```

**Test**: Forward pass, verify shapes at each stage, count parameters

#### 2.6 Create ByteFormer Configs
- `configs/byteformer_jpeg.yaml` (JPEG only)
- `configs/byteformer_png.yaml` (PNG only)
- `configs/byteformer_mixed.yaml` (Mixed format)

#### 2.7 Train ByteFormer Variants
```bash
# Train JPEG-only
uv run python scripts/train_byteformer.py --config configs/byteformer_jpeg.yaml

# Train PNG-only
uv run python scripts/train_byteformer.py --config configs/byteformer_png.yaml

# Train mixed
uv run python scripts/train_byteformer.py --config configs/byteformer_mixed.yaml
```

**Expected Results** (per variant):
- Training time: ~15-20 hours (100 epochs, A100)
- Target accuracy: 80-90% on test set
- Memory usage: Higher than ResNet-18

#### 2.8 Compare Formats
Create `scripts/compare_formats.py`:
```python
# Load checkpoints for JPEG, PNG, Mixed
# Test each on JPEG test set
# Test each on PNG test set
# Create comparison table
```

**Analysis**:
- Does mixed training improve cross-format generalization?
- Which format is easier to learn from?
- Is there a format-specific accuracy gap?

### Success Criteria
- ✅ ByteFormer trains successfully
- ✅ Achieves ≥80% test accuracy (at least one variant)
- ✅ Mixed format training works
- ✅ Format comparison completed

### Deliverables
- `src/data/cifar10_bytes.py`
- `src/data/transforms.py`
- `src/models/byteformer.py`
- `src/models/layers.py`
- `scripts/train_byteformer.py`
- `scripts/analyze_byte_lengths.py`
- `scripts/compare_formats.py`
- `configs/byteformer_{jpeg,png,mixed}.yaml`
- `results/byteformer_jpeg.json`
- `results/byteformer_png.json`
- `results/byteformer_mixed.json`
- `checkpoints/byteformer_{jpeg,png,mixed}_best.pt`

---

## Phase 3: Robustness Evaluation (Days 11-14)

### Goals
- Test robustness to byte corruption
- Test format generalization
- Demonstrate advantages of byte-level encoding

### Tasks

#### 3.1 Implement Corruption Utils (`src/data/corruption.py`)
```python
class ByteCorruption:
    def __init__(self, corruption_rate=0.01):
        pass

    def __call__(self, byte_tensor):
        # Randomly flip bytes
        pass

class PixelNoiseCorruption:
    def __init__(self, noise_std=0.1):
        pass

    def __call__(self, image_tensor):
        # Add Gaussian noise to pixels
        pass

class FileTruncation:
    def __init__(self, truncation_ratio=0.1):
        pass

    def __call__(self, byte_tensor):
        # Zero out trailing bytes
        pass
```

#### 3.2 Test Byte Corruption Robustness
Create `scripts/test_byte_corruption.py`:
```python
# Load best ByteFormer checkpoint (mixed)
# Load best ResNet-18 checkpoint
# Test ByteFormer with byte corruption (0.1%, 0.5%, 1%, 2%, 5%)
# Test ResNet-18 with pixel noise (equivalent levels)
# Plot accuracy vs corruption rate
```

**Hypothesis**: Byte-level models more robust to byte corruption

#### 3.3 Test Format Generalization
Create `scripts/test_format_generalization.py`:
```python
# Convert CIFAR-10 test set to WebP, BMP (if feasible)
# Test ByteFormer (mixed) on new formats
# Test ResNet-18 on same formats (decoded pixels)
# Compare zero-shot performance
```

**Hypothesis**: Mixed training enables better format generalization

#### 3.4 Test File Truncation
Create `scripts/test_truncation.py`:
```python
# Load ByteFormer checkpoint
# Test with trailing bytes removed (5%, 10%, 20%, 50%)
# Plot accuracy vs truncation level
```

**Analysis**: How gracefully does performance degrade?

#### 3.5 Test JPEG Quality Robustness
Create `scripts/test_jpeg_quality.py`:
```python
# Train ByteFormer on JPEG quality 75
# Test on quality [50, 60, 70, 80, 90, 100]
# Train ResNet-18 on quality 75
# Test on same range
# Compare robustness
```

#### 3.6 Create Comprehensive Comparison
Create `scripts/create_comparison_table.py`:
```python
# Load all results
# Create markdown table with:
#   - Model, Accuracy, FLOPs, Params, Training Time
#   - Robustness metrics
# Generate plots
```

### Success Criteria
- ✅ At least 1 robustness test shows byte-level advantage
- ✅ All experiments documented
- ✅ Comparison table created

### Deliverables
- `src/data/corruption.py`
- `scripts/test_byte_corruption.py`
- `scripts/test_format_generalization.py`
- `scripts/test_truncation.py`
- `scripts/test_jpeg_quality.py`
- `scripts/create_comparison_table.py`
- `results/robustness_comparison.json`
- `results/comparison_table.md`
- `notebooks/03_results_visualization.ipynb`

---

## Phase 4: Advanced Models (Optional, Days 15-21)

### Goals
- Implement ViT-Tiny baseline
- Implement hybrid byte+pixel model
- Explore architectural improvements

### Tasks

#### 4.1 Implement ViT-Tiny (`src/models/vit.py`)
```python
class ViTTinyCIFAR(nn.Module):
    def __init__(self, image_size=32, patch_size=4, d_model=192,
                 depth=12, heads=3):
        pass
```

**Train**: 100 epochs on CIFAR-10

**Expected**: 85-90% accuracy (ViT underperforms CNNs on small datasets)

#### 4.2 Implement Hybrid Model (`src/models/hybrid.py`)
```python
class HybridBytePixelModel(nn.Module):
    def __init__(self, byte_encoder, pixel_encoder, fusion_dim=384):
        self.byte_branch = byte_encoder  # ByteFormer
        self.pixel_branch = pixel_encoder  # ResNet-18 backbone
        self.fusion = nn.Sequential(...)  # Fusion MLP
        pass

    def forward(self, byte_seq, pixel_image):
        # Process both modalities
        # Fuse representations
        # Classify
        pass
```

#### 4.3 Train Hybrid Model
```bash
uv run python scripts/train_hybrid.py --config configs/hybrid_model.yaml
```

**Hypothesis**: Hybrid model exceeds both modalities alone

#### 4.4 Explore Architectural Variations
- Try different downsampling ratios (4:1, 16:1)
- Try different window sizes (64, 256)
- Try sparse attention patterns
- Try state space models (Mamba) instead of attention

**Goal**: Find optimal architecture for byte sequences

### Success Criteria
- ✅ Hybrid model implemented and trained
- ✅ At least one architectural variation tested
- ✅ Results documented

### Deliverables
- `src/models/vit.py`
- `src/models/hybrid.py`
- `scripts/train_hybrid.py`
- `configs/hybrid_model.yaml`
- `results/hybrid_results.json`
- `results/architectural_ablations.json`

---

## Phase 5: Documentation and Analysis (Days 22-23)

### Goals
- Create comprehensive README
- Document findings
- Prepare reproducible artifacts

### Tasks

#### 5.1 Create README.md
```markdown
# Byte-Level Image Encoding for CIFAR-10

## Overview
...

## Installation
uv sync
...

## Quick Start
...

## Results
...

## Citation
...
```

#### 5.2 Create RESULTS.md
Document:
- Final accuracy for all models
- Computational costs
- Robustness analysis
- Key findings
- Limitations
- Future work

#### 5.3 Create Reproducibility Guide
- `REPRODUCING.md` with exact commands
- Seed settings
- Hardware requirements
- Expected outputs

#### 5.4 Clean Up Code
- Add docstrings
- Run linters (ruff, black)
- Add type hints
- Write unit tests

#### 5.5 Create Visualizations
- Training curves
- Robustness plots
- Architecture diagrams
- Attention visualizations (if possible)

### Deliverables
- `README.md`
- `RESULTS.md`
- `REPRODUCING.md`
- Clean, documented codebase
- Visualization notebooks

---

## Reproducibility Checklist

### Fixed Seeds
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Hyperparameter Documentation
All configs saved as YAML files with:
- Model architecture details
- Training hyperparameters
- Data augmentation settings
- Random seeds

### Checkpoint Management
- Save best model (based on validation accuracy)
- Save final model
- Save optimizer state
- Save training history
- Include config in checkpoint

### Logging
- Use tensorboard or wandb for metric tracking
- Log every 100 iterations
- Save training curves as images
- Log system info (GPU, PyTorch version, etc.)

---

## Resource Estimates

### Computational Requirements

| Phase | GPU Hours (A100) | Wall Time | Storage |
|-------|-----------------|-----------|---------|
| Phase 0 | 0 | 1 hour | 500 MB |
| Phase 1 | 3 | 1 day | 1 GB |
| Phase 2 | 60 | 3-4 days | 5 GB |
| Phase 3 | 10 | 1-2 days | 2 GB |
| Phase 4 | 30 | 2-3 days | 3 GB |
| **Total** | **~103** | **~2 weeks** | **~11 GB** |

### Parallelization Opportunities
- Train different format variants in parallel (3 jobs)
- Run robustness tests in parallel (4 jobs)
- Architectural ablations in parallel (variable)

**With 3-4 GPUs**: Reduce wall time to ~1 week

---

## Risk Mitigation

### Risk 1: ByteFormer Doesn't Train
**Symptoms**: Loss doesn't decrease, accuracy stays at random (10%)
**Mitigations**:
- Reduce learning rate
- Increase batch size
- Simplify architecture (fewer layers)
- Check gradients (use `torch.autograd.grad_check`)

### Risk 2: Out of Memory
**Symptoms**: CUDA OOM errors
**Mitigations**:
- Reduce batch size
- Reduce max_length (8192 → 4096)
- Use gradient accumulation
- Enable mixed precision training (fp16)

### Risk 3: Accuracy Too Low
**Symptoms**: ByteFormer <70% accuracy
**Mitigations**:
- Train longer (100 → 200 epochs)
- Tune hyperparameters (learning rate, weight decay)
- Try different downsampling strategies
- Verify data pipeline (check byte sequences are correct)

### Risk 4: No Robustness Advantage
**Symptoms**: Byte models not more robust than pixel models
**Mitigations**:
- Try different corruption types
- Test on more diverse scenarios
- Focus on format generalization (clearer advantage)
- Document negative results honestly

---

## Success Metrics Summary

### Minimum Viable Product (MVP)
1. ✅ ResNet-18 baseline: ≥92% accuracy
2. ✅ ByteFormer: ≥80% accuracy
3. ✅ Training completes successfully
4. ✅ Basic comparison documented

### Target Goals
5. ✅ ByteFormer: ≥85% accuracy
6. ✅ At least 1 robustness advantage demonstrated
7. ✅ Format comparison completed
8. ✅ Comprehensive documentation

### Stretch Goals
9. ⏳ Hybrid model: ≥95% accuracy
10. ⏳ Architectural ablations completed
11. ⏳ Multiple robustness advantages shown
12. ⏳ Published on GitHub with reproducible artifacts

---

## Daily Checklist Template

### Daily Progress Tracking
- [ ] What did I complete today?
- [ ] What challenges did I encounter?
- [ ] What are tomorrow's priorities?
- [ ] Do I need to adjust the plan?

### Weekly Review
- [ ] Are we on track?
- [ ] What's the accuracy so far?
- [ ] Any blockers?
- [ ] Do we need to pivot?

---

## Final Deliverables

### Code
- Clean, documented Python codebase
- Configuration files for all experiments
- Training and evaluation scripts
- Unit tests

### Documentation
- CLAUDE.md (project overview)
- RESEARCH.md (literature review)
- ENCODING_SCHEMES.md (approach definitions)
- DESIGN.md (architecture specifications)
- IMPLEMENTATION.md (this document)
- README.md (user guide)
- RESULTS.md (findings)
- REPRODUCING.md (reproducibility guide)

### Data
- Trained model checkpoints
- Training logs and metrics
- Result JSON files
- Comparison tables

### Visualizations
- Training curves
- Robustness plots
- Architecture diagrams
- Attention maps (if feasible)

### Analysis
- Jupyter notebooks with analysis
- Comparison tables (markdown)
- Summary statistics

---

## Next Steps

After completing this plan:

1. **Write Paper/Blog Post**: Summarize findings for broader audience
2. **Open Source**: Publish on GitHub with reproducible artifacts
3. **Scale Up**: Try on ImageNet or other datasets
4. **Extend to Other Tasks**: Segmentation, detection, generation
5. **Optimize Further**: Efficient architectures, quantization, distillation
6. **Explore Applications**: Privacy-preserving inference, format-agnostic models

---

*Last updated: 2025-11-21*
