# Byte-Level Image Encoding Research

> **Exploring models that operate directly on image file bytes rather than pixel arrays**

## 🎯 Project Status

**Phase**: Planning Complete ✅
**Next Step**: Implementation (Phase 0 - Project Setup)

## 📋 Overview

This research project investigates whether computer vision models can effectively process images at the byte level (raw file bytes from PNG/JPEG) instead of the traditional pixel-level representation. The goal is to:

1. **Systematically compare** byte-level vs. pixel-level encoding
2. **Measure performance** on CIFAR-10 classification
3. **Test robustness** to corruption, format changes, and compression artifacts
4. **Explore unified** text-image tokenization via bytes

## 🔬 Research Questions

1. Can byte-level models achieve competitive accuracy with pixel-level models?
2. Are byte-level models more robust to file corruption and format variations?
3. What is the computational cost tradeoff (accuracy vs. FLOPs)?
4. Can mixed-format training improve format generalization?

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Project overview and goals |
| [RESEARCH.md](RESEARCH.md) | Comprehensive literature review |
| [ENCODING_SCHEMES.md](ENCODING_SCHEMES.md) | Encoding approaches and comparisons |
| [DESIGN.md](DESIGN.md) | Technical architecture and design decisions |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Phased implementation plan with timelines |

## 🏗️ Planned Architecture

### Baseline Models
- **ResNet-18**: CNN baseline on pixels (expected: 92-95% accuracy)
- **ViT-Tiny**: Transformer baseline on patches (expected: 85-90% accuracy)

### Byte-Level Models
- **ByteFormer-CIFAR**: Transformer operating on file bytes
  - Byte embedding (256 vocab)
  - Strided convolution downsampling (8:1)
  - Shifted window attention (window size 128)
  - Hierarchical downsampling (32:1 total)
  - Target: 80-90% accuracy

### Advanced Models (Phase 4)
- **Hybrid Model**: Combines byte + pixel representations

## 📊 Planned Experiments

### Phase 1: Baseline (Days 2-4)
- Train ResNet-18 on CIFAR-10 pixels
- Establish performance benchmark

### Phase 2: Byte-Level (Days 5-10)
- Implement byte-level data pipeline
- Train ByteFormer on JPEG, PNG, and mixed formats
- Compare cross-format performance

### Phase 3: Robustness (Days 11-14)
- Test byte corruption robustness
- Test format generalization (WebP, BMP)
- Test file truncation
- Test JPEG quality robustness

### Phase 4: Advanced (Days 15-21, Optional)
- Implement hybrid byte+pixel model
- Explore architectural variations

## 🔑 Key Findings from Literature Review

1. **ByteFormer (Apple, 2023)** demonstrated 77.3% ImageNet accuracy processing TIFF file bytes
2. **EvaByte (2025)** showed vision-language capabilities with byte-level JPEG processing
3. **Main Challenge**: Byte sequences are 48-62× longer than patch tokens
4. **Computational Cost**: ~18× more FLOPs than pixel-based ViT
5. **Format Sensitivity**: TIFF/PNG easier than JPEG (Huffman encoding complexity)

## 💡 Novel Contributions

This project will provide:

1. **First systematic comparison** of byte vs. pixel encodings on same dataset/budget
2. **Format robustness analysis** with mixed-format training
3. **Comprehensive efficiency evaluation** (accuracy per FLOP)
4. **Open-source reproducible implementation**

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: uv
- **Framework**: PyTorch 2.0+
- **Dataset**: CIFAR-10 (60K images, 32×32 RGB, 10 classes)
- **Compute**: NVIDIA GPU (A100 recommended)

## 📈 Success Criteria

### Minimum Viable Product
- ✅ ResNet-18 achieves ≥92% accuracy
- ✅ ByteFormer trains successfully
- ✅ ByteFormer achieves ≥80% accuracy
- ✅ Basic comparison documented

### Target Goals
- 🎯 ByteFormer ≥85% accuracy
- 🎯 Demonstrate at least 1 robustness advantage
- 🎯 Complete format comparison experiments
- 🎯 Comprehensive documentation

### Stretch Goals
- 🌟 Hybrid model ≥95% accuracy
- 🌟 Multiple robustness advantages shown
- 🌟 Published on GitHub with full reproducibility

## 📖 Quick Start (After Implementation)

```bash
# Clone repository
git clone <repo-url>
cd byte_level_image_encoding

# Set up environment
uv sync

# Train baseline
uv run python scripts/train_resnet18.py --config configs/resnet18_pixels.yaml

# Train ByteFormer
uv run python scripts/train_byteformer.py --config configs/byteformer_mixed.yaml

# Run robustness tests
uv run python scripts/test_robustness.py

# Analyze results
uv run python scripts/analyze_results.py
```

## 📊 Expected Results

| Model | Accuracy | FLOPs | Params | Training Time |
|-------|----------|-------|--------|---------------|
| ResNet-18 | 92-95% | 0.56G | 11M | ~3 hrs |
| ViT-Tiny | 85-90% | 0.30G | 5M | ~1.5 hrs |
| ByteFormer (JPEG) | 80-85% | 2.5G | 9M | ~17 hrs |
| ByteFormer (PNG) | 80-85% | 2.5G | 9M | ~17 hrs |
| ByteFormer (Mixed) | 83-88% | 2.5G | 9M | ~17 hrs |
| Hybrid | 93-95% | 3.0G | 20M | ~20 hrs |

*Estimates based on literature review and computational analysis*

## 🎓 References

1. **ByteFormer**: Buch et al. (2023) - "Bytes Are All You Need: Transformers Operating Directly On File Bytes" ([arXiv:2306.00238](https://arxiv.org/abs/2306.00238))

2. **EvaByte**: HKU + SambaNova (2025) - "Efficient Byte-level Language Models at Scale" ([Blog](https://hkunlp.github.io/blog/2025/evabyte/))

3. **Byte Latent Transformer**: Meta AI (2024) - "Byte Latent Transformer: Patches Scale Better Than Tokens" ([arXiv:2412.09871](https://arxiv.org/abs/2412.09871))

4. **ByT5**: Xue et al. (2021) - "ByT5: Towards a Token-Free Future" ([arXiv:2105.13626](https://arxiv.org/abs/2105.13626))

5. **Perceiver IO**: Jaegle et al. (2021) - "Perceiver IO: A General Architecture for Structured Inputs & Outputs" ([arXiv:2107.14795](https://arxiv.org/abs/2107.14795))

## 🤝 Contributing

This is a research prototype. Contributions welcome after initial implementation is complete.

## 📄 License

TBD

## 👥 Authors

Research prototype developed as part of AI research exploration.

---

**Status**: Planning phase complete. Ready to begin implementation.

*Last updated: 2025-11-21*
