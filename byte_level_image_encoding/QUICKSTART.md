# Quick Start Guide: Format-Agnostic Zero-Shot Transfer

**Objective**: Train ByteFormer on JPEG images, test zero-shot on PNG/WebP/BMP

**Timeline**: 1-2 weeks (part-time) or 2-3 days (full-time)

**Compute**: ~20 GPU hours (RTX 3090 / A100)

---

## Prerequisites

- **Hardware**: GPU with ≥16GB VRAM (RTX 3090 / A100)
- **Software**: Python 3.8+, CUDA 11.7+
- **Storage**: ~20 GB free space

---

## Step-by-Step Instructions

### Phase 0: Environment Setup (30 minutes)

```bash
# 1. Navigate to project directory
cd byte_level_image_encoding

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output**:
```
PyTorch: 2.0.0+cu117
CUDA: True
```

---

### Phase 1: Data Preparation (2-4 hours)

#### Step 1.1: Download CIFAR-10

```bash
python scripts/download_cifar10.py
```

**Expected output**:
```
Download Complete!
Training samples: 50,000
Test samples: 10,000
```

**Storage**: ~170 MB

---

#### Step 1.2: Encode to Multiple Formats

```bash
python src/data/encode_formats.py
```

**Expected output**:
```
ENCODING COMPLETE - SUMMARY
Format     Avg Size     Total Files     Total Size
----------------------------------------------------------
JPEG       1500 bytes   60,000 files    85.8 MB
PNG        1200 bytes   60,000 files    68.7 MB
WEBP       800 bytes    60,000 files    45.8 MB
BMP        3078 bytes   60,000 files    176.3 MB
```

**Duration**: 1-2 hours
**Storage**: ~400 MB total

---

#### Step 1.3: Test Data Loading

```bash
# Test byte dataset loader
python src/data/byte_dataset.py data/cifar10/jpeg_q75/train
```

**Expected output**:
```
ByteImageDataset(
  data_dir=data/cifar10/jpeg_q75/train,
  num_files=50000,
  num_classes=10,
  max_bytes=8192,
  avg_file_size=1500 bytes
)
Test passed!
```

---

### Phase 2: Model Testing (15 minutes)

```bash
# Test ByteFormer model
python src/models/byteformer.py
```

**Expected output**:
```
Model Configuration:
  model_name: ByteFormer
  d_model: 192
  nhead: 6
  num_layers: 6
  total_parameters: 8,234,506

Test passed!
```

---

### Phase 3: Training (12-15 GPU hours)

#### Option A: Quick Test (10 epochs, ~1 hour)

```bash
python src/training/train.py --config quick_test
```

**Expected**: ~40-50% accuracy (under-trained, for testing only)

---

#### Option B: Full Training (100 epochs, 12-15 hours)

```bash
# Start training on JPEG only
python src/training/train.py --config jpeg_q75

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir experiments/h1_zero_shot_transfer/jpeg_q75/logs
```

**Expected progress**:
```
Epoch 1/100:
  Train Loss: 2.1234, Train Acc: 25.43%
  Test Loss: 1.9876, Test Acc: 28.12%

Epoch 50/100:
  Train Loss: 0.4321, Train Acc: 85.67%
  Test Loss: 0.5234, Test Acc: 82.45%
  ✓ New best accuracy: 82.45%

Epoch 100/100:
  Train Loss: 0.2134, Train Acc: 92.34%
  Test Loss: 0.4876, Test Acc: 84.23%

Training Complete!
Best Test Accuracy: 84.23%
Total Time: 14.5 hours
```

**Checkpoints saved to**: `experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints/`

---

### Phase 4: Zero-Shot Evaluation (30 minutes)

```bash
# Evaluate on all formats
python src/training/evaluate.py \
  --checkpoint experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints/best_model.pth \
  --formats jpeg_q75 png webp bmp
```

**Expected output** (example scenario):
```
ZERO-SHOT TRANSFER ANALYSIS
======================================================================
Baseline (trained on jpeg_q75): 84.23%

Format          Accuracy      Transfer Ratio   Interpretation
----------------------------------------------------------------------
jpeg_q75        84.23%        100.0%           (in-distribution)
png             73.45%         87.2%           ✓ Strong transfer
webp            68.12%         80.9%           ✓ Strong transfer
bmp             71.89%         85.3%           ✓ Strong transfer

HYPOTHESIS 1 (H1) EVALUATION
======================================================================
H1: Model achieves >50% of in-distribution accuracy on zero-shot formats

  png            :  87.2%  ✓ PASS
  webp           :  80.9%  ✓ PASS
  bmp            :  85.3%  ✓ PASS

----------------------------------------------------------------------
  🎉 H1 CONFIRMED: Format-agnostic learning detected!
  → Model learned content, not just format-specific patterns
```

**Results saved to**: `experiments/h1_zero_shot_transfer/results/zero_shot_results.json`

---

### Phase 5: Visualization (10 minutes)

```bash
# Generate publication-quality plots
python src/analysis/visualize.py \
  --results experiments/h1_zero_shot_transfer/results/zero_shot_results.json \
  --output-dir experiments/h1_zero_shot_transfer/results/figures
```

**Output**:
```
Visualizations saved to: experiments/h1_zero_shot_transfer/results/figures
Generated figures:
  1. accuracy_by_format.png - Overall accuracy comparison
  2. transfer_ratios.png - Transfer performance vs H1 threshold
  3. per_class_heatmap.png - Per-class accuracy matrix
  4. class_transfer_analysis.png - Which classes transfer best
```

---

## Interpreting Results

### Scenario A: Strong Transfer (≥70%)

**Finding**: Format-agnostic learning! 🎉

**Interpretation**: ByteFormer learned semantic content (cats, planes, etc.), not just format-specific byte patterns (JPEG headers, Huffman codes).

**Publication potential**: CVPR/ICCV (major positive result)

**Implication**: Train once on any format, deploy on all formats

---

### Scenario B: Partial Transfer (40-70%)

**Finding**: Mixed content + format learning

**Interpretation**: Model learned both semantic features AND format-specific patterns. Need to analyze:
- Which classes transfer best?
- Which formats are most similar?
- Does file size affect transfer?

**Publication potential**: CVPR/ICCV/NeurIPS (analysis contribution)

**Implication**: Need better architectures to disentangle content from format

---

### Scenario C: Weak Transfer (≤40%)

**Finding**: Format-specific learning

**Interpretation**: ByteFormer heavily relies on format-specific byte patterns (JPEG DCT coefficients, PNG scanline filters). Semantic content not primary feature.

**Publication potential**: NeurIPS (important negative result)

**Implication**: Fundamental limitation of naive byte-level approach. Need format-invariant preprocessing or multi-format training.

---

## Troubleshooting

### Out of GPU Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python src/training/train.py --config jpeg_q75 --batch-size 32
```

---

### Training Too Slow

**Solution**: Enable mixed precision (should be on by default)

Check `src/training/config.py`:
```python
mixed_precision: bool = True  # Should be True
```

---

### Poor JPEG Accuracy (<70%)

**Causes**:
- Under-training (train longer)
- Learning rate too high/low
- Model too small

**Solution**:
```bash
# Train for 150 epochs
python src/training/train.py --config jpeg_q75 --epochs 150
```

---

### All Formats Same Accuracy

**Problem**: Data encoding failed, all formats identical

**Verification**:
```bash
# Check file headers are different
hexdump -C data/cifar10/jpeg_q75/test/00000_class3.jpeg | head -n 2
hexdump -C data/cifar10/png/test/00000_class3.png | head -n 2
hexdump -C data/cifar10/bmp/test/00000_class3.bmp | head -n 2
```

**Expected**: Different byte patterns (JPEG starts with `FFD8`, PNG with `8950`, BMP with `424D`)

---

## Next Steps

### 1. Analyze Results

- Which classes transfer best? (vehicles vs animals?)
- Does transfer quality correlate with file size?
- Are there consistent failure modes?

### 2. Ablation Studies

- Train on PNG, test on JPEG (reverse direction)
- Train on mixed formats (50% JPEG, 50% PNG)
- Vary JPEG quality (Q50, Q75, Q95)

### 3. Advanced Experiments

- Attention visualization (where does model look?)
- Byte pattern statistics (which bytes matter?)
- Pixel-level baseline (ResNet-18 on decoded images)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/download_cifar10.py` | Download CIFAR-10 |
| `src/data/encode_formats.py` | Encode to JPEG/PNG/WebP/BMP |
| `src/data/byte_dataset.py` | Load files as byte sequences |
| `src/models/byteformer.py` | ByteFormer architecture |
| `src/training/config.py` | Training configurations |
| `src/training/train.py` | Training loop |
| `src/training/evaluate.py` | Zero-shot evaluation |
| `src/analysis/visualize.py` | Generate plots |

---

## Directory Structure After Completion

```
byte_level_image_encoding/
├── data/
│   ├── raw/                          # Original CIFAR-10
│   └── cifar10/
│       ├── jpeg_q75/                 # 60K JPEG files (~86 MB)
│       ├── png/                      # 60K PNG files (~69 MB)
│       ├── webp/                     # 60K WebP files (~46 MB)
│       └── bmp/                      # 60K BMP files (~176 MB)
├── experiments/
│   └── h1_zero_shot_transfer/
│       ├── jpeg_q75/
│       │   ├── checkpoints/          # Model checkpoints
│       │   │   ├── best_model.pth    # Best model
│       │   │   └── final_model.pth
│       │   ├── logs/                 # TensorBoard logs
│       │   ├── config.json           # Training config
│       │   └── training_summary.json
│       └── results/
│           ├── zero_shot_results.json  # Evaluation results
│           └── figures/                # Publication plots
│               ├── accuracy_by_format.png
│               ├── transfer_ratios.png
│               ├── per_class_heatmap.png
│               └── class_transfer_analysis.png
└── [source code files]
```

---

## Estimated Timeline

| Phase | Task | Duration | GPU Hours |
|-------|------|----------|-----------|
| 0 | Setup | 30 min | 0 |
| 1 | Data prep | 2-4 hrs | 0 |
| 2 | Model test | 15 min | 0 |
| 3 | Training | 12-15 hrs | 12-15 |
| 4 | Evaluation | 30 min | 0.5 |
| 5 | Visualization | 10 min | 0 |
| **Total** | | **15-20 hrs** | **~15** |

**Part-time** (4 hrs/day): 1-2 weeks
**Full-time** (8 hrs/day): 2-3 days (mostly waiting for training)

---

## Success Checklist

- [ ] CIFAR-10 downloaded (170 MB)
- [ ] All 4 formats encoded (60K files each)
- [ ] ByteFormer trains without errors
- [ ] JPEG test accuracy ≥80%
- [ ] Zero-shot evaluation runs on all formats
- [ ] Results JSON saved
- [ ] Visualizations generated
- [ ] Interpretation drafted

---

## Support

**Issues?** Check `IMPLEMENTATION.md` for detailed troubleshooting.

**Questions?** Review `RESEARCH.md` for background on hypothesis.

**Code bugs?** Check individual file docstrings for usage examples.

---

*Last updated: 2025-11-22*
*Ready for implementation!*
