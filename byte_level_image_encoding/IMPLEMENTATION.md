# Implementation Plan: Format-Agnostic Zero-Shot Transfer

**Project**: Byte-Level Image Encoding Research
**Primary Objective**: Test if byte-level models learn format-agnostic or format-specific representations
**Timeline**: 2-3 weeks (part-time) or 1 week (full-time)
**Compute Budget**: ~20 GPU hours (RTX 3090 / A100)
**Status**: Ready to implement

---

## Executive Summary

This implementation plan focuses on the **primary novel research question**: Can a byte-level model trained on JPEG bytes classify PNG/WebP/BMP images zero-shot?

**Experiment Design**:
1. Encode CIFAR-10 in 4 formats (JPEG, PNG, WebP, BMP)
2. Train ByteFormer on JPEG-only
3. Test zero-shot on all 4 formats
4. Analyze results: format-agnostic vs format-specific learning

**Expected Outcome**: Publishable result regardless of transfer success/failure

---

## Repository Structure

```
byte_level_image_encoding/
├── README.md
├── CLAUDE.md
├── RESEARCH.md
├── IMPLEMENTATION.md (this file)
├── data/
│   ├── cifar10/
│   │   ├── jpeg_q75/  # CIFAR-10 as JPEG quality 75
│   │   ├── png/       # CIFAR-10 as PNG
│   │   ├── webp/      # CIFAR-10 as WebP
│   │   └── bmp/       # CIFAR-10 as BMP
│   └── raw/           # Original CIFAR-10
├── src/
│   ├── data/
│   │   ├── encode_formats.py
│   │   └── byte_dataset.py
│   ├── models/
│   │   └── byteformer.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── config.py
│   └── analysis/
│       ├── visualize.py
│       └── failure_analysis.py
├── experiments/
│   └── h1_zero_shot_transfer/
│       ├── jpeg_q75/
│       └── results/
└── requirements.txt
```

---

## Phase 0: Environment Setup (Day 1, ~2 hours)

### Dependencies

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

### Installation

```bash
cd byte_level_image_encoding
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Hardware Requirements

**Minimum**: RTX 3090 (24GB VRAM), 32 GB RAM, 20 GB storage
**Optimal**: A100 (40GB), 64 GB RAM, 50 GB storage

---

## Phase 1: Data Preparation (Day 1-2, ~4 hours)

### Step 1.1: Download CIFAR-10

```python
# scripts/download_cifar10.py
import torchvision

trainset = torchvision.datasets.CIFAR10(root='./data/raw', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False, download=True)

print(f"Train: {len(trainset)}, Test: {len(testset)}")
```

### Step 1.2: Encode to Multiple Formats

Create `src/data/encode_formats.py` to convert CIFAR-10 to JPEG/PNG/WebP/BMP.

**Expected file sizes** (32×32 images):
- JPEG (Q75): ~1,500 bytes
- PNG: ~1,200 bytes
- WebP (Q75): ~800 bytes
- BMP: ~3,078 bytes

### Step 1.3: Byte Dataset Loader

Create `src/data/byte_dataset.py`:

```python
class ByteImageDataset(Dataset):
    """Load image files as raw byte sequences"""
    def __init__(self, data_dir, max_bytes=8192, pad_value=0):
        self.data_dir = Path(data_dir)
        self.max_bytes = max_bytes
        self.files = sorted(list(self.data_dir.glob('*.*')))
        self.labels = [int(f.stem.split('_class')[1]) for f in self.files]

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            byte_values = list(f.read())

        # Pad or truncate to max_bytes
        if len(byte_values) > self.max_bytes:
            byte_values = byte_values[:self.max_bytes]
        else:
            byte_values += [0] * (self.max_bytes - len(byte_values))

        return torch.tensor(byte_values, dtype=torch.long), \
               torch.tensor(self.labels[idx], dtype=torch.long)
```

---

## Phase 2: Model Implementation (Day 2-3, ~6 hours)

### ByteFormer Architecture

Create `src/models/byteformer.py`:

```python
class ByteFormer(nn.Module):
    """ByteFormer for CIFAR-10"""
    def __init__(
        self,
        max_bytes=8192,
        d_model=192,
        nhead=6,
        num_layers=6,
        downsample_stride=16,
        num_classes=10
    ):
        super().__init__()

        # 1. Byte embedding (256 vocab)
        self.byte_embedding = nn.Embedding(256, d_model, padding_idx=0)

        # 2. Downsampling (strided conv)
        self.downsample = nn.Conv1d(d_model, d_model,
                                     kernel_size=32, stride=16)

        # 3. Positional encoding
        seq_len = math.ceil(max_bytes / downsample_stride)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=768,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 5. Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, byte_sequence):
        # Embed + downsample + transformer + classify
        x = self.byte_embedding(byte_sequence)
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)
```

**Expected**: ~8-10M parameters

---

## Phase 3: Training (Day 3-5, ~12-15 GPU hours)

### Training Configuration

Create `src/training/config.py`:

```python
@dataclass
class TrainingConfig:
    # Model
    max_bytes: int = 8192
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 6

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01

    # Paths
    data_dir: str = './data/cifar10'
    output_dir: str = './experiments/h1_zero_shot_transfer'
```

### Training Script

Create `src/training/train.py` with standard PyTorch training loop.

**Expected**:
- Training time: 12-15 hours (RTX 3090)
- JPEG test accuracy: 80-85%
- GPU memory: ~18-20 GB

### Training Command

```bash
python src/training/train.py
```

---

## Phase 4: Zero-Shot Evaluation (Day 5-6, ~2 hours)

### Evaluation Script

Create `src/training/evaluate.py`:

```python
def main():
    # Load model trained on JPEG
    model = load_model('./experiments/h1_zero_shot_transfer/jpeg_q75/best_model.pth')

    # Evaluate on all formats
    formats = ['jpeg_q75', 'png', 'webp', 'bmp']
    results = {}

    for fmt in formats:
        acc = evaluate_format(model, fmt)
        results[fmt] = acc

    # Save results
    save_results(results)

    # Print summary
    jpeg_acc = results['jpeg_q75']
    print(f"JPEG (train): {jpeg_acc:.2f}%")
    for fmt in ['png', 'webp', 'bmp']:
        ratio = results[fmt] / jpeg_acc
        print(f"{fmt}: {results[fmt]:.2f}% ({ratio:.1%} transfer)")
```

### Evaluation Command

```bash
python src/training/evaluate.py
```

---

## Phase 5: Analysis & Visualization (Day 6-7, ~4 hours)

### Visualizations

Create `src/analysis/visualize.py`:

1. **Accuracy by format** (bar chart)
2. **Transfer ratios** (PNG/WebP/BMP relative to JPEG)
3. **Per-class heatmap** (which classes transfer well?)

### Failure Analysis

Create `src/analysis/failure_analysis.py`:

- Which classes transfer best/worst?
- Do simpler classes (e.g., vehicles) transfer better than complex (e.g., animals)?
- Format-specific patterns (BMP > PNG > WebP)?

---

## Results Interpretation

### Scenario A: High Transfer (≥70%)

**Finding**: Format-agnostic learning!
**Publication**: CVPR/ICCV (major positive result)
**Implication**: Train once, deploy anywhere

### Scenario B: Zero Transfer (≤30%)

**Finding**: Format-specific learning!
**Publication**: NeurIPS (important negative result)
**Implication**: Fundamental limitation of byte-level approach

### Scenario C: Partial Transfer (40-70%)

**Finding**: Mixed content + format features
**Publication**: CVPR/ICCV/NeurIPS (analysis contribution)
**Implication**: Need better architectures to separate content from format

---

## Success Criteria

### Must-Have
- [x] CIFAR-10 encoded in 4 formats
- [ ] ByteFormer trained on JPEG (≥80% accuracy)
- [ ] Zero-shot evaluation complete
- [ ] Results visualizations created
- [ ] Draft interpretation document

### Should-Have
- [ ] Per-class transfer analysis
- [ ] Attention map visualization
- [ ] Byte pattern statistics

### Nice-to-Have
- [ ] Pixel-level baseline comparison
- [ ] Mixed-format training experiment
- [ ] Hybrid pixel-byte model

---

## Timeline Summary

| Phase | Duration | GPU Hours | Status |
|-------|----------|-----------|--------|
| 0. Setup | 2 hrs | 0 | ⏳ |
| 1. Data Prep | 4 hrs | 0 | ⏳ |
| 2. Model | 6 hrs | 0 | ⏳ |
| 3. Training | 2 days | 12-15 | ⏳ |
| 4. Eval | 2 hrs | 0.5 | ⏳ |
| 5. Analysis | 4 hrs | 0 | ⏳ |
| 6. Writing | 3 days | 0 | ⏳ |
| **TOTAL** | **1-2 weeks** | **~15-20** | |

---

## Troubleshooting

**Out of GPU memory**: Reduce batch_size (64→32)
**Training slow**: Use mixed precision (torch.cuda.amp)
**Poor JPEG accuracy**: Train longer (100→150 epochs)
**All formats same accuracy**: Verify formats actually different (check file headers)

---

## Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Prepare data
python src/data/encode_formats.py

# 3. Train
python src/training/train.py

# 4. Evaluate
python src/training/evaluate.py

# 5. Visualize
python src/analysis/visualize.py
```

---

*Last updated: 2025-11-22*
*Status: Ready for implementation*
