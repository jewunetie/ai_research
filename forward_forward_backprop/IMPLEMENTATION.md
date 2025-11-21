# Implementation Plan: Forward-Forward with Backprop MVP

## Executive Summary

This document outlines the implementation plan for a **Minimum Viable Product (MVP)** that explores hybrid training schemes combining Forward-Forward (FF) and Backpropagation (BP).

**Target Hardware**: M4 Max MacBook Pro (MPS backend)
**Success Criteria**: Match or improve BP accuracy + generate interesting insights
**Scope**: MVP with clear extension path to comprehensive study

---

## Table of Contents
1. [MVP Architecture Overview](#mvp-architecture-overview)
2. [Implementation Approaches](#implementation-approaches)
3. [Baseline Comparisons](#baseline-comparisons)
4. [Project Structure](#project-structure)
5. [Technical Specifications](#technical-specifications)
6. [Training Procedures](#training-procedures)
7. [Evaluation and Analysis](#evaluation-and-analysis)
8. [Extension Path](#extension-path)

---

## MVP Architecture Overview

### Core Network Architecture

**Base Network** (for MNIST/Fashion-MNIST):
```
Input (28×28 = 784)
    ↓
[Layer 1: FC(784 → 500) + ReLU]  ← Can use FF or BP
    ↓
[Layer 2: FC(500 → 500) + ReLU]  ← Can use FF or BP
    ↓
[Layer 3: FC(500 → 500) + ReLU]  ← Can use FF or BP
    ↓
[Classifier: FC(500 → 10)]        ← Always BP
    ↓
Output (10 classes)
```

**Design Rationale**:
- 3 hidden layers: Standard for FF experiments (Hinton 2022)
- 500 neurons: Large enough to learn, small enough for M4 Max
- Simple fully-connected: Easier to analyze than CNNs
- ReLU activation: Standard for both FF and BP

**Scaling Path**: Can easily swap to larger architectures or CNNs for CIFAR-10

---

## Implementation Approaches

### Approach 1: Sequential Phased (Primary Focus)

**Three-phase training pipeline**:

#### Phase 1: FF Pretraining (Unsupervised)
```python
# Train layers 1-3 with Forward-Forward
for epoch in range(100):
    for batch in dataloader:
        # Create positive samples (original data)
        positive = batch

        # Create negative samples (augmented/corrupted data)
        negative = augment(batch)  # Random label, pixel shuffle, etc.

        # FF training for each layer
        for layer in ff_layers:
            h_pos = layer(positive)
            h_neg = layer(negative)

            goodness_pos = (h_pos ** 2).sum(1)
            goodness_neg = (h_neg ** 2).sum(1)

            # Threshold-based loss
            loss = -torch.log(1 + torch.exp(-(goodness_pos - threshold))).mean() \
                   -torch.log(1 + torch.exp(goodness_neg - threshold)).mean()

            loss.backward()
            optimizer.step()

            # Normalize for next layer
            positive = layer(positive).detach() / layer(positive).detach().norm(dim=1, keepdim=True)
            negative = layer(negative).detach() / layer(negative).detach().norm(dim=1, keepdim=True)
```

#### Phase 2a: Frozen FF + BP Classifier
```python
# Freeze FF layers
for param in ff_layers.parameters():
    param.requires_grad = False

# Train classifier only
for epoch in range(50):
    for batch, labels in dataloader:
        # Forward through frozen FF layers
        h = ff_layers(batch)

        # BP training on classifier
        logits = classifier(h)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

#### Phase 2b: Fine-tune All with BP
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# End-to-end BP fine-tuning
for epoch in range(50):
    for batch, labels in dataloader:
        logits = model(batch)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

**Key Variables to Test**:
- Number of FF layers (1, 2, or 3)
- Pretraining epochs (50, 100, 200)
- Whether to do Phase 2b (fine-tuning) or just Phase 2a (frozen)
- Negative sample generation strategy

---

### Approach 2: Detached Interface (Novel Contribution)

**Simultaneous FF+BP training**:

```python
# FF layers continuously train with FF
# BP layers continuously train with BP
# Gradient detachment at interface

for epoch in range(150):
    for batch in dataloader:
        # === FF Training (unsupervised) ===
        positive = batch
        negative = augment(batch)

        h_pos = ff_layers(positive)
        h_neg = ff_layers(negative)

        goodness_pos = (h_pos ** 2).sum(1)
        goodness_neg = (h_neg ** 2).sum(1)

        ff_loss = compute_ff_threshold_loss(goodness_pos, goodness_neg)
        ff_loss.backward()
        ff_optimizer.step()

        # === BP Training (supervised) ===
        with torch.no_grad():
            h_detached = ff_layers(batch).detach()  # Stop gradients

        logits = bp_layers(h_detached)
        bp_loss = cross_entropy(logits, labels)
        bp_loss.backward()  # Only updates BP layers
        bp_optimizer.step()
```

**Key Innovation**:
- FF and BP train simultaneously
- `.detach()` cleanly separates gradient flow
- FF layers maintain local learning properties
- BP layers optimize for supervised task

**Analysis Questions**:
- Do FF representations improve over time for the supervised task?
- How does the interface evolve during training?
- Can we visualize what FF learns vs what BP needs?

---

### Approach 3: Block-wise Hybrid (Validation)

**Implementation for comparison with SFF paper**:

```python
# Divide network into blocks
blocks = [
    [layer1, layer2],  # Block 1
    [layer3],           # Block 2
]

# Each block has auxiliary classifier
aux_classifiers = [
    nn.Linear(500, 10),  # For block 1
    nn.Linear(500, 10),  # For block 2
]

# Training
for epoch in range(150):
    for batch, labels in dataloader:
        h = batch
        total_loss = 0

        for block, aux_clf in zip(blocks, aux_classifiers):
            # BP within block
            for layer in block:
                h = layer(h)

            # Local auxiliary loss
            aux_logits = aux_clf(h)
            aux_loss = cross_entropy(aux_logits, labels)

            total_loss += aux_loss

            # Detach between blocks (FF-style)
            h = h.detach()

        # Backprop each block's loss separately
        total_loss.backward()
        optimizer.step()
```

**Purpose**: Replicate SFF findings as validation

---

## Baseline Comparisons

### Baseline 1: Pure Backpropagation
```python
# Standard end-to-end training
model = MLP([784, 500, 500, 500, 10])

for epoch in range(100):
    for batch, labels in dataloader:
        logits = model(batch)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

**Expected Performance**:
- MNIST: ~98-99%
- Fashion-MNIST: ~88-90%

### Baseline 2: Pure Forward-Forward
```python
# All layers trained with FF
# Final layer uses goodness for classification

for epoch in range(200):  # FF typically needs more epochs
    for batch in dataloader:
        # FF training for all layers
        ff_train_all_layers(batch)

# Classification via goodness per class
def classify(x):
    goodness_per_class = []
    for class_label in range(10):
        x_with_label = embed_label(x, class_label)
        h = model(x_with_label)
        goodness = (h ** 2).sum(1)
        goodness_per_class.append(goodness)
    return torch.argmax(torch.stack(goodness_per_class), dim=0)
```

**Expected Performance**:
- MNIST: ~97-98% (slightly worse than BP)
- Fashion-MNIST: ~85-87%

### Baseline 3: Random Init + BP Classifier
```python
# No pretraining, just train classifier on random features
ff_layers = MLP([784, 500, 500, 500])
ff_layers.requires_grad_(False)  # Keep random

classifier = nn.Linear(500, 10)

for epoch in range(50):
    for batch, labels in dataloader:
        h = ff_layers(batch)
        logits = classifier(h)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

**Expected Performance**:
- Very poor (~20-40% on MNIST)
- Establishes value of pretraining

### Baseline 4: Autoencoder Pretrain + BP Fine-tune
```python
# Phase 1: Autoencoder pretraining
encoder = MLP([784, 500, 500, 500])
decoder = MLP([500, 500, 500, 784])

for epoch in range(100):
    for batch in dataloader:
        h = encoder(batch)
        reconstructed = decoder(h)
        loss = mse_loss(reconstructed, batch)
        loss.backward()
        optimizer.step()

# Phase 2: BP fine-tuning
classifier = nn.Linear(500, 10)
model = nn.Sequential(encoder, classifier)

for epoch in range(50):
    for batch, labels in dataloader:
        logits = model(batch)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
```

**Purpose**: Standard unsupervised pretraining baseline

---

## Project Structure

```
forward_forward_backprop/
├── CLAUDE.md                          # Project overview
├── RESEARCH.md                        # Literature review
├── INTERMIXED_ARCHITECTURES_RESEARCH.md  # Deep dive on hybrids
├── IMPLEMENTATION.md                  # This file
├── pyproject.toml                     # uv project configuration
├── README.md                          # Getting started guide
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ff_layer.py               # Forward-Forward layer implementation
│   │   ├── mlp.py                     # Modular MLP architectures
│   │   └── hybrid_models.py           # Hybrid FF+BP models
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ff_trainer.py              # FF training logic
│   │   ├── bp_trainer.py              # Standard BP training
│   │   ├── sequential_phased.py       # Sequential phased training
│   │   ├── detached_interface.py      # Detached interface training
│   │   └── block_wise.py              # Block-wise hybrid training
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py                # Dataset loaders
│   │   └── augmentation.py            # Negative sample generation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, loss tracking
│   │   ├── visualization.py           # t-SNE, training curves
│   │   └── analysis.py                # Representation quality analysis
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging.py                 # Experiment logging
│       └── device.py                  # MPS/CPU device handling
│
├── experiments/
│   ├── configs/                       # YAML config files
│   │   ├── mnist_baseline_bp.yaml
│   │   ├── mnist_sequential_phased.yaml
│   │   ├── mnist_detached_interface.yaml
│   │   └── fashion_mnist_*.yaml
│   │
│   ├── run_experiment.py              # Main experiment runner
│   ├── run_baselines.py               # Run all baselines
│   └── compare_results.py             # Compare and visualize results
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_explore_data.ipynb
│   ├── 02_visualize_representations.ipynb
│   └── 03_analyze_results.ipynb
│
├── results/                           # Experiment outputs
│   ├── logs/                          # Training logs
│   ├── checkpoints/                   # Model checkpoints
│   ├── figures/                       # Plots and visualizations
│   └── metrics/                       # CSV/JSON metrics
│
└── tests/
    ├── test_ff_layer.py
    ├── test_training.py
    └── test_models.py
```

---

## Technical Specifications

### Hardware Optimization for M4 Max

```python
import torch

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
```

**Memory Considerations**:
- M4 Max has unified memory (RAM = VRAM)
- Batch size: 512-1024 for MNIST (small)
- Batch size: 256-512 for CIFAR-10
- Larger models possible due to unified memory

**Performance Optimizations**:
- Use `torch.compile()` if available (PyTorch 2.0+)
- Mixed precision training with `torch.autocast()` for MPS
- Efficient data loading with num_workers=4-8

### Dependencies (pyproject.toml)

```toml
[project]
name = "forward-forward-backprop"
version = "0.1.0"
description = "Hybrid FF+BP training research"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "wandb>=0.15.0",  # Optional: experiment tracking
    "pyyaml>=6.0",
    "tensorboard>=2.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.0.280",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Configuration System

**Example config (YAML)**:
```yaml
# experiments/configs/mnist_sequential_phased.yaml

experiment:
  name: "mnist_sequential_phased_3layers"
  seed: 42
  device: "auto"  # auto-detect MPS/CUDA/CPU

data:
  dataset: "mnist"
  batch_size: 512
  num_workers: 4

model:
  architecture: "mlp"
  hidden_dims: [500, 500, 500]
  activation: "relu"
  num_classes: 10

training:
  approach: "sequential_phased"

  # Phase 1: FF Pretraining
  phase1:
    num_ff_layers: 3
    epochs: 100
    learning_rate: 0.03
    threshold: 2.0
    negative_strategy: "random_label"  # or "augment", "shuffle"

  # Phase 2a: Frozen FF + BP Classifier
  phase2a:
    freeze_ff: true
    epochs: 50
    learning_rate: 0.001
    optimizer: "adam"

  # Phase 2b: Fine-tune All
  phase2b:
    enabled: true
    epochs: 50
    learning_rate: 0.0001
    optimizer: "adam"

evaluation:
  metrics: ["accuracy", "loss"]
  save_checkpoints: true
  visualization: true
  linear_probing: true  # Test representation quality

logging:
  use_tensorboard: true
  use_wandb: false
  log_interval: 10  # batches
```

---

## Training Procedures

### Forward-Forward Training Details

**Goodness Function**:
```python
def compute_goodness(h):
    """
    Compute goodness as sum of squared activations.

    Args:
        h: Layer activations [batch_size, hidden_dim]

    Returns:
        goodness: Scalar goodness value [batch_size]
    """
    return (h ** 2).sum(dim=1)
```

**Threshold Loss**:
```python
def ff_threshold_loss(goodness_pos, goodness_neg, threshold=2.0):
    """
    Forward-Forward loss with threshold.

    Positive samples should have goodness > threshold
    Negative samples should have goodness < threshold
    """
    loss_pos = -torch.log(1 + torch.exp(-(goodness_pos - threshold))).mean()
    loss_neg = -torch.log(1 + torch.exp(goodness_neg - threshold)).mean()
    return loss_pos + loss_neg
```

**Normalization Between Layers**:
```python
def normalize_layer_output(h):
    """
    Normalize activations to prevent trivial solutions.

    Each sample normalized to unit length.
    """
    return h / (h.norm(dim=1, keepdim=True) + 1e-8)
```

**Negative Sample Generation Strategies**:

1. **Random Label (Supervised)**:
```python
def generate_negative_labels(labels, num_classes=10):
    """Random wrong labels"""
    negative_labels = torch.randint(0, num_classes, labels.shape)
    # Ensure different from positive
    mask = negative_labels == labels
    negative_labels[mask] = (negative_labels[mask] + 1) % num_classes
    return negative_labels
```

2. **Data Augmentation (Unsupervised)**:
```python
def generate_negative_augmented(images):
    """Strong augmentation as negatives"""
    # Random noise
    noise = torch.randn_like(images) * 0.3
    return torch.clamp(images + noise, 0, 1)
```

3. **Pixel Shuffle**:
```python
def generate_negative_shuffled(images):
    """Shuffle pixels within each image"""
    batch_size = images.shape[0]
    shuffled = images.view(batch_size, -1)
    for i in range(batch_size):
        perm = torch.randperm(shuffled.shape[1])
        shuffled[i] = shuffled[i, perm]
    return shuffled.view_as(images)
```

### Learning Rate Schedules

**For FF Pretraining**:
```python
# Higher learning rate, cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=0.001
)
```

**For BP Fine-tuning**:
```python
# Lower learning rate, step decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)
```

---

## Evaluation and Analysis

### Primary Metrics

1. **Test Accuracy**: Main performance metric
2. **Training Loss**: Convergence behavior
3. **Training Time**: Epochs and wall-clock time
4. **Memory Usage**: Peak memory during training

### Representation Quality Analysis

**Linear Probing**:
```python
def linear_probe_layer(model, layer_idx, train_loader, test_loader):
    """
    Freeze model up to layer_idx, train linear classifier on top.
    Tests representation quality of that layer.
    """
    # Extract features
    features_train, labels_train = extract_features(model, layer_idx, train_loader)
    features_test, labels_test = extract_features(model, layer_idx, test_loader)

    # Train linear classifier
    probe = nn.Linear(features_train.shape[1], 10)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

    for epoch in range(50):
        logits = probe(features_train)
        loss = F.cross_entropy(logits, labels_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        logits = probe(features_test)
        accuracy = (logits.argmax(1) == labels_test).float().mean()

    return accuracy.item()
```

**t-SNE Visualization**:
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_representations(model, layer_idx, dataloader, title):
    """
    Visualize learned representations with t-SNE.
    """
    features, labels = extract_features(model, layer_idx, dataloader)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.cpu().numpy())

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels.cpu().numpy(), cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(f'results/figures/{title.replace(" ", "_")}.png')
    plt.close()
```

**Centered Kernel Alignment (CKA)**:
```python
def compute_cka(features_a, features_b):
    """
    Compute CKA similarity between two sets of representations.
    Higher CKA = more similar representations.
    """
    # Center features
    features_a = features_a - features_a.mean(0)
    features_b = features_b - features_b.mean(0)

    # Compute kernels
    kernel_a = features_a @ features_a.T
    kernel_b = features_b @ features_b.T

    # CKA
    hsic = (kernel_a * kernel_b).sum()
    norm_a = (kernel_a * kernel_a).sum() ** 0.5
    norm_b = (kernel_b * kernel_b).sum() ** 0.5

    return hsic / (norm_a * norm_b)
```

### Comparison Table

After all experiments, generate comparison table:

```python
results = {
    "Method": [
        "Pure BP",
        "Pure FF",
        "Random Init + BP",
        "Autoencoder + BP",
        "Sequential Phased (Frozen)",
        "Sequential Phased (Fine-tuned)",
        "Detached Interface",
        "Block-wise Hybrid"
    ],
    "MNIST Accuracy": [...],
    "Fashion-MNIST Accuracy": [...],
    "Training Time (min)": [...],
    "Peak Memory (GB)": [...],
    "Layer 1 Probe Acc": [...],
    "Layer 2 Probe Acc": [...],
    "Layer 3 Probe Acc": [...]
}

df = pd.DataFrame(results)
print(df.to_markdown())
df.to_csv('results/comparison_table.csv')
```

---

## Extension Path

### MVP → Comprehensive Study

**Phase 1: MVP (Current)**
- ✅ MNIST + Fashion-MNIST
- ✅ Sequential Phased approach
- ✅ Detached Interface approach
- ✅ 4 baselines
- ✅ Basic analysis (accuracy, linear probing, t-SNE)

**Phase 2: Scaling Up**
- Add CIFAR-10 dataset
- Add CNN architectures
- Test different split points (1, 2, 3 FF layers)
- More negative generation strategies

**Phase 3: Deep Analysis**
- Layer-wise CKA analysis
- Gradient flow visualization
- Ablation studies (threshold, learning rates, epochs)
- Sample efficiency curves (training with 10%, 50%, 100% data)

**Phase 4: Novel Variations**
- True layer-by-layer alternating (FF→BP→FF→BP)
- Learned split points (meta-learning)
- Adaptive threshold in FF
- Hybrid loss (FF + BP simultaneously without detach)

**Phase 5: Publication**
- Write paper
- Clean code release
- Reproducibility package
- Extended benchmarks

---

## Success Criteria

### Must Have (MVP)
- ✅ All implementations run without errors
- ✅ Baseline BP achieves expected accuracy (>98% MNIST)
- ✅ Sequential Phased matches or improves BP
- ✅ Detached Interface trains successfully
- ✅ Clear comparison table and visualizations

### Should Have
- Sequential Phased improves over Random Init + BP (validates pretraining)
- Detached Interface shows interesting representation evolution
- Linear probing reveals layer-wise representation quality differences
- t-SNE visualizations show clear class separation

### Nice to Have
- Sequential Phased improves over pure BP
- Detached Interface matches BP accuracy
- Evidence for when/why FF pretraining helps
- Novel insights about FF representations

---

## Implementation Checklist

### Core Components
- [ ] FF Layer implementation with goodness computation
- [ ] Negative sample generation (3 strategies)
- [ ] Sequential Phased trainer (3 phases)
- [ ] Detached Interface trainer
- [ ] Pure BP baseline
- [ ] Pure FF baseline
- [ ] Random Init baseline
- [ ] Autoencoder baseline
- [ ] Block-wise hybrid trainer

### Infrastructure
- [ ] Dataset loaders (MNIST, Fashion-MNIST)
- [ ] Configuration system (YAML)
- [ ] Experiment runner
- [ ] Metrics logging (TensorBoard)
- [ ] Checkpoint saving/loading
- [ ] Device handling (MPS/CPU)

### Analysis
- [ ] Training curve plotting
- [ ] Accuracy comparison table
- [ ] Linear probing evaluation
- [ ] t-SNE visualization
- [ ] CKA similarity analysis

### Documentation
- [ ] README with getting started
- [ ] Code comments
- [ ] Example configs
- [ ] Results interpretation guide

---

## Timeline Estimate

### Week 1: Core Implementation
- Days 1-2: Project setup, models, FF layer
- Days 3-4: Sequential Phased trainer
- Days 5-7: Baselines + Detached Interface

### Week 2: Experiments & Analysis
- Days 1-3: Run all experiments on MNIST
- Days 4-5: Run experiments on Fashion-MNIST
- Days 6-7: Analysis, visualization, comparison

### Week 3: Extension (Optional)
- Scale to CIFAR-10
- Additional experiments based on findings
- Paper writing / documentation

---

## Next Steps

1. **Set up project structure** with uv
2. **Implement core FF layer** with goodness computation
3. **Implement Sequential Phased trainer** (main approach)
4. **Implement baselines** for comparison
5. **Run initial experiments** on MNIST
6. **Analyze results** and iterate

Let's start coding! 🚀

---

**Last Updated**: November 2025
**Author**: Claude Code
**Project**: Forward-Forward with Backprop Research
