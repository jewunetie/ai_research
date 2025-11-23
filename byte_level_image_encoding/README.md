# Byte-Level Image Encoding: Format-Agnostic Zero-Shot Transfer

> **🔥 Novel Research: Can models trained on JPEG classify PNG/WebP/BMP zero-shot?**

## 🎯 Project Status

**Phase**: ✅ Implementation Complete
**Next Step**: Run experiments (see [QUICKSTART.md](QUICKSTART.md))
**Timeline**: 1-2 weeks | ~20 GPU hours

## 🚀 What Makes This Novel?

This project explores an **untested research question**: Do byte-level vision models learn **format-agnostic** or **format-specific** representations?

### The Experiment (H1)

Train ByteFormer on **JPEG-only** → Test zero-shot on **PNG/WebP/BMP**

**Why this matters:**
- ✅ **Never tested before**: ByteFormer paper trained/tested separate configs per format
- ✅ **Minimal compute**: 20 GPU hours on CIFAR-10 (not ImageNet!)
- ✅ **All outcomes publishable**:
  - Strong transfer (≥70%): Format-agnostic learning → **CVPR/ICCV**
  - Partial transfer (40-70%): Mixed representations → **Analysis paper**
  - Weak transfer (≤40%): Format-specific learning → **Important negative result**

## 📋 Quick Start

**Prerequisites**: Python 3.8+, GPU with ≥16GB VRAM, ~20GB storage

```bash
# 1. Setup environment
cd byte_level_image_encoding
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download & encode data (~4 hours)
python scripts/download_cifar10.py
python src/data/encode_formats.py

# 3. Train on JPEG only (~15 GPU hours)
python src/training/train.py --config jpeg_q75

# 4. Zero-shot evaluation (~30 min)
python src/training/evaluate.py \
  --checkpoint experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints/best_model.pth \
  --formats jpeg_q75 png webp bmp

# 5. Generate plots (~10 min)
python src/analysis/visualize.py \
  --results experiments/h1_zero_shot_transfer/results/zero_shot_results.json
```

**Full instructions**: See [QUICKSTART.md](QUICKSTART.md)

## 🔬 Research Question

**Hypothesis (H1)**: A byte-level model trained on JPEG will achieve **>50% of its in-distribution accuracy** when tested zero-shot on PNG/WebP/BMP.

### Why This Hasn't Been Done

From our literature review (14 targeted searches, Nov 2024):

1. **ByteFormer (Apple 2023)**: GitHub shows separate training configs for TIFF/PNG/JPEG. No cross-format evaluation.
2. **EvaByte (HKU 2025)**: JPEG only. No format generalization tested.
3. **bGPT (Microsoft 2024)**: Showed cross-*modal* transfer (images→audio), but not cross-*format*.
4. **Community evidence**: Stack Overflow user reported 20% accuracy drop (JPEG→PNG) but no controlled study.

**Gap**: File format as a domain shift factor has never been systematically studied in byte-level models.

## 📊 Expected Outcomes

### Scenario A: Strong Transfer (≥70%)
```
JPEG:  84% accuracy (in-distribution)
PNG:   73% accuracy (87% transfer ratio) ✓
WebP:  68% accuracy (81% transfer ratio) ✓
BMP:   72% accuracy (86% transfer ratio) ✓
```
**Interpretation**: Model learned semantic content, not just JPEG byte patterns!
**Publication**: CVPR/ICCV (major positive result)

### Scenario B: Partial Transfer (40-70%)
```
JPEG:  84% accuracy
PNG:   50% accuracy (60% transfer ratio)
WebP:  46% accuracy (55% transfer ratio)
BMP:   52% accuracy (62% transfer ratio)
```
**Interpretation**: Mixed content + format-specific features
**Publication**: NeurIPS/ICLR (analysis contribution)

### Scenario C: Weak Transfer (≤40%)
```
JPEG:  84% accuracy
PNG:   25% accuracy (30% transfer ratio) ✗
WebP:  22% accuracy (26% transfer ratio) ✗
BMP:   28% accuracy (33% transfer ratio) ✗
```
**Interpretation**: Model relies heavily on format-specific patterns
**Publication**: NeurIPS (important negative result)

## 🏗️ Architecture

**ByteFormer** (~8-10M parameters):
- Byte embedding (256 vocab)
- Strided 1D conv downsampling (8192 → 513 tokens)
- 6-layer transformer encoder (d_model=192, nhead=6)
- Global average pooling + classifier

**Training**: 100 epochs, batch_size=64, AdamW, cosine LR schedule
**Expected JPEG accuracy**: 80-85%

## 📁 Repository Structure

```
byte_level_image_encoding/
├── README.md                     # This file
├── QUICKSTART.md                 # Step-by-step guide
├── RESEARCH.md                   # Literature review + H1 details
├── IMPLEMENTATION.md             # Technical implementation plan
├── requirements.txt              # Python dependencies
├── scripts/
│   └── download_cifar10.py       # Download CIFAR-10
├── src/
│   ├── data/
│   │   ├── encode_formats.py     # Encode to JPEG/PNG/WebP/BMP
│   │   └── byte_dataset.py       # Load files as byte sequences
│   ├── models/
│   │   └── byteformer.py         # ByteFormer architecture
│   ├── training/
│   │   ├── config.py             # Training configurations
│   │   ├── train.py              # Training loop
│   │   └── evaluate.py           # Zero-shot evaluation
│   └── analysis/
│       └── visualize.py          # Generate plots
└── experiments/                  # Results go here
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Step-by-step implementation guide |
| [RESEARCH.md](RESEARCH.md) | Literature review + H1 hypothesis |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Technical details + timeline |
| [DESIGN.md](DESIGN.md) | Original architecture decisions |
| [PIVOTS.md](PIVOTS.md) | Pivot research (10 alternative directions) |

## 🔑 Key Technical Details

### File Format Differences (32×32 CIFAR-10 image)

| Format | Compression | Color Space | Byte Structure | Size |
|--------|-------------|-------------|----------------|------|
| JPEG (Q75) | DCT + Huffman | YCbCr | Non-aligned codes | ~1.5 KB |
| PNG | Deflate (LZSS) | RGB | Scanline filtering | ~1.2 KB |
| WebP (Q75) | VP8 codec | YUV | Intra-frame prediction | ~800 B |
| BMP | None | RGB | Uncompressed bitmap | ~3.1 KB |

**Question**: Can ByteFormer trained on JPEG's DCT+Huffman patterns recognize PNG's Deflate-compressed bytes?

## 🎓 References

### Primary References (Byte-Level Vision)
1. **ByteFormer**: Buch et al. (2023) - "Bytes Are All You Need" ([arXiv:2306.00238](https://arxiv.org/abs/2306.00238))
2. **EvaByte**: HKU + SambaNova (2025) - "Efficient Byte-level Language Models" ([Blog](https://hkunlp.github.io/blog/2025/evabyte/))
3. **bGPT**: Microsoft (2024) - "Beyond Language Models: Byte Models are Digital World Simulators" ([arXiv:2402.19155](https://arxiv.org/abs/2402.19155))

### Related Work (Cross-Modal Transfer)
4. **ByteNet**: Oct 2024 - "Rethinking Multimedia File Fragment Classification" ([arXiv:2410.20855](https://arxiv.org/abs/2410.20855))
5. **Hendrycks et al.** (2020) - "Measuring Robustness to Natural Distribution Shifts" (NeurIPS 2020)

See [RESEARCH.md](RESEARCH.md) for full literature review.

## 💡 Why This Experiment is Valuable

### If H1 Passes (Strong Transfer)
- **Implication**: Train once on any format, deploy on all formats
- **Impact**: Simplifies data pipelines (no format conversion)
- **Follow-up**: Does this generalize to ImageNet? Other modalities?

### If H1 Fails (Weak Transfer)
- **Implication**: Byte models learn format-specific patterns, not semantic content
- **Impact**: Need format-invariant preprocessing or multi-format training
- **Follow-up**: Can we design architectures that ignore format?

### Either Way
- **Contribution**: First systematic study of format as domain shift
- **Open questions**: Which classes transfer? Does file size matter? Attention visualization?

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch 2.0+ (with AMP)
- **Dataset**: CIFAR-10 (60K images, 10 classes)
- **Compute**: RTX 3090 / A100 (~20 GPU hours)
- **Storage**: ~20 GB

## 📈 Success Criteria

✅ **Implementation Complete**:
- [x] Data pipeline (download + multi-format encoding)
- [x] ByteFormer model (~8M params)
- [x] Training loop (mixed precision, checkpointing)
- [x] Zero-shot evaluation script
- [x] Visualization pipeline

🎯 **Experiment Goals**:
- [ ] JPEG accuracy ≥80%
- [ ] Zero-shot evaluation on 4 formats
- [ ] H1 validation (transfer ratio analysis)
- [ ] Publication-quality plots
- [ ] Results interpretation

🌟 **Stretch Goals**:
- [ ] Attention visualization (which bytes matter?)
- [ ] Ablation: Train on PNG, test JPEG (reverse)
- [ ] Mixed-format training
- [ ] Per-class transfer analysis

## 🤝 Next Steps

1. **Run Experiment** (1-2 weeks):
   - Follow [QUICKSTART.md](QUICKSTART.md)
   - Monitor training with TensorBoard
   - Evaluate zero-shot transfer

2. **Analyze Results**:
   - Check H1 pass/fail criteria
   - Identify which classes transfer best/worst
   - Look for patterns in failure modes

3. **Write Paper** (Optional):
   - Use generated plots
   - Discuss implications
   - Target: CVPR/ICCV/NeurIPS

## 📄 License

TBD

## 👥 Authors

Research prototype exploring format-agnostic zero-shot transfer in byte-level vision models.

---

**Status**: ✅ Implementation complete, ready to run
**Last updated**: 2025-11-22

**Get Started**: See [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions.
