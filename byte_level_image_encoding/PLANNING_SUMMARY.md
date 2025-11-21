# Planning Summary: Byte-Level Image Encoding

**Date**: 2025-11-21
**Status**: ✅ Planning Complete → Ready for Implementation

---

## Executive Summary

This document summarizes the completed planning phase for a research project investigating **byte-level image encoding** - where models process raw image file bytes (JPEG/PNG) instead of decoded pixel arrays.

**Key Decision**: Focus on **CIFAR-10** with **conservative downsampling** (8:1) and **mixed format training** (JPEG + PNG) to balance feasibility with scientific rigor.

---

## Planning Process Completed

### ✅ Step 1: Project Initialization
- Created project directory: `byte_level_image_encoding/`
- Defined research goals and scope in `CLAUDE.md`
- Established dataset (CIFAR-10), stack (Python, uv, PyTorch), and objectives

### ✅ Step 2: Quick Feasibility Check
- Searched for: byte-level image models, multimodal byte transformers, ByT5 extensions
- **Key Finding**: ByteFormer (Apple, 2023) achieved 77.3% ImageNet accuracy on file bytes
- **Verdict**: Feasible but computationally challenging (18× more FLOPs than ViT)

### ✅ Step 3: Deep Research Phase
- Analyzed 6 major works: ByteFormer, EvaByte, BLT, PixelBytes, Perceiver IO, Unified Multimodal
- Identified sequence length as primary challenge (48-62× longer than patches)
- Found format sensitivity (TIFF: 77%, JPEG: 66% for ByteFormer)
- Documented computational requirements and architectural solutions

### ✅ Step 4: Research Summary Documentation
- Created comprehensive `RESEARCH.md` (11 sections, 5000+ words)
- Analyzed prior work, sequence lengths, challenges, advantages, and gaps
- Identified 4 testable hypotheses for experiments
- Listed recommended reading and key references

### ✅ Step 5: Feasibility Assessment
**Questions Answered**:
1. ✅ Has byte-level been tried? **Yes** (ByteFormer, EvaByte)
2. ✅ Main bottlenecks? **Sequence length, compute, format complexity**
3. ✅ Can we make it tractable? **Yes, with CIFAR-10 and smart architectures**
4. ✅ Testable hypothesis? **Yes, 4 concrete hypotheses defined**

**Decision**: Proceed with conservative approach optimized for learning and iteration

### ✅ Step 6: Encoding Schemes Defined
- Created `ENCODING_SCHEMES.md` with 3 main approaches:
  - **Baseline 1A**: ResNet-18 on pixels (92-95% expected)
  - **Baseline 1B**: ViT-Tiny on patches (85-90% expected)
  - **Approach 2A**: ByteFormer-CIFAR on bytes (80-90% target)
  - **Approach 2C**: Hybrid byte+pixel (future work)
- Defined sequence length comparison (65 tokens → 8,192 bytes)
- Specified 4 robustness evaluation schemes

### ✅ Step 7: Implementation Design
- Created `DESIGN.md` with complete architecture specifications:
  - **Data Pipeline**: Byte-level and pixel-level datasets
  - **ByteFormer Architecture**: Embedding → Downsample (8:1) → Shifted Windows → Hierarchical → Classify
  - **Shifted Window Attention**: 1D variant to handle long sequences
  - **Training Pipeline**: Unified trainer for all models
- Addressed 6 open questions (padding, downsampling, windows, format mixing, etc.)
- Defined success criteria (must-have, should-have, nice-to-have)

### ✅ Step 8: Open Questions Addressed
All design decisions finalized:
- Padding: Zeros (0x00)
- Downsampling: 8:1 (conservative)
- Window size: 128 tokens
- Format mixing: 50/50 JPEG/PNG
- Validation: Use standard test set
- Target: Within 10% + robustness advantage

### ✅ Step 9: Implementation Plan Created
- Created detailed `IMPLEMENTATION.md` with 5 phases:
  - **Phase 0**: Project setup (Day 1)
  - **Phase 1**: Baseline implementation (Days 2-4)
  - **Phase 2**: Byte-level implementation (Days 5-10)
  - **Phase 3**: Robustness evaluation (Days 11-14)
  - **Phase 4**: Advanced models - optional (Days 15-21)
  - **Phase 5**: Documentation and analysis (Days 22-23)
- Estimated resources: ~103 GPU hours (A100), 2-3 weeks wall time
- Defined reproducibility checklist and risk mitigation strategies

### ✅ Step 10: Documentation Complete
- Created `README.md` as project landing page
- All planning documents cross-referenced
- Ready for public consumption or team collaboration

---

## Key Design Decisions

### Dataset: CIFAR-10
**Rationale**:
- Small images (32×32) → manageable byte sequences (3-10 KB)
- Well-established benchmark
- Fast iteration for experimentation
- Proven baseline (ResNet-18: 92-95%)

### Downsampling: Conservative 8:1
**Rationale**:
- Balances information preservation with compute
- Initial: 8,192 bytes → 1,024 tokens (8:1)
- Hierarchical: 1,024 → 512 → 256 (4:1 total)
- Final: 32:1 overall ratio (8,192 → 256 tokens)

### Format: Mixed Training (50% JPEG / 50% PNG)
**Rationale**:
- Tests format generalization hypothesis
- No format-specific bias
- Enables robustness evaluation
- Mirrors real-world diversity

### Target Accuracy: Within 10% + Robustness
**Rationale**:
- Realistic given computational constraints (4-5× slower than ResNet)
- Focus on unique advantages (robustness, format agnostic)
- ByteFormer on ImageNet was 77% vs ViT 72% (closer gap expected on CIFAR)
- Pragmatic: 80-85% byte vs 92-95% pixel is valuable if robustness shown

---

## Architecture Specifications

### ByteFormer-CIFAR
```
Input: 8,192 bytes (LongTensor)
  ↓
Byte Embedding: 256 vocab → 192 dim
  ↓
Positional Encoding: Learned (8,192 × 192)
  ↓
Strided Conv 1D: 8:1 downsampling → 1,024 tokens
  ↓
Transformer Block 0-1: Shifted Window Attention (window=128)
  ↓
Downsample 2:1 → 512 tokens
  ↓
Transformer Block 2-3: Shifted Window Attention
  ↓
Downsample 2:1 → 256 tokens
  ↓
Transformer Block 4-5: Shifted Window Attention
  ↓
Global Average Pooling → 192 dim
  ↓
Linear: 192 → 10 classes
```

**Parameters**: ~9M (comparable to ResNet-18's 11M)
**FLOPs**: ~2.5 GFLOPs per image (~4-5× ResNet-18's 0.56G)

---

## Planned Experiments

### Experiment 1: Baseline vs Byte-Level
**Comparison**: ResNet-18 (pixels) vs ByteFormer (bytes)
- Accuracy: ?
- FLOPs: 0.56G vs 2.5G
- Training time: 3 hrs vs 17 hrs
- **Metric**: Accuracy per FLOP

### Experiment 2: Format Comparison
**Variants**: ByteFormer trained on JPEG-only, PNG-only, Mixed
- Test on JPEG test set
- Test on PNG test set
- **Hypothesis**: Mixed training improves cross-format accuracy

### Experiment 3: Byte Corruption Robustness
**Setup**: Randomly flip bytes at rates [0.1%, 0.5%, 1%, 2%, 5%]
- ByteFormer (Mixed) with byte corruption
- ResNet-18 with pixel noise (equivalent levels)
- **Hypothesis**: Byte models more robust to byte corruption

### Experiment 4: Format Generalization
**Setup**: Train on {JPEG, PNG}, test on {WebP, BMP, TIFF}
- Zero-shot evaluation (no fine-tuning)
- **Hypothesis**: Byte models generalize better across formats

### Experiment 5: File Truncation
**Setup**: Remove trailing bytes [5%, 10%, 20%, 50%]
- **Hypothesis**: Byte models gracefully degrade with partial files

### Experiment 6: JPEG Quality Robustness
**Setup**: Train on quality 75, test on [50, 60, 70, 80, 90, 100]
- Compare ByteFormer vs ResNet-18 robustness curve
- **Hypothesis**: Byte models robust to compression artifacts

---

## Success Metrics

### Phase 1 Success (Baseline)
- [ ] ResNet-18 trains successfully
- [ ] Achieves ≥92% accuracy
- [ ] Baseline documented (accuracy, FLOPs, time)

### Phase 2 Success (Byte-Level)
- [ ] ByteFormer trains without errors
- [ ] Achieves ≥80% accuracy (at least one variant)
- [ ] Mixed format training works
- [ ] Format comparison completed

### Phase 3 Success (Robustness)
- [ ] At least 1 robustness test shows byte-level advantage
- [ ] All experiments run and documented
- [ ] Comparison table created

### Phase 4 Success (Advanced, Optional)
- [ ] Hybrid model implemented and trained
- [ ] ViT-Tiny baseline completed
- [ ] Architectural ablations documented

---

## Resource Requirements

### Computational
- **Phase 1**: 3 GPU hours (ResNet-18)
- **Phase 2**: 60 GPU hours (3 ByteFormer variants × 20 hrs each)
- **Phase 3**: 10 GPU hours (robustness tests)
- **Phase 4**: 30 GPU hours (hybrid + ablations)
- **Total**: ~103 GPU hours (A100 equivalent)

### Storage
- CIFAR-10 dataset: 170 MB
- Model checkpoints: ~5 GB
- Logs and results: ~2 GB
- Byte-converted datasets: ~4 GB
- **Total**: ~11 GB

### Time
- Sequential execution: ~2-3 weeks
- Parallel execution (3-4 GPUs): ~1 week
- Documentation: +3 days

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ByteFormer doesn't train | Medium | High | Reduce LR, simplify architecture, verify gradients |
| Out of memory | Medium | High | Reduce batch size, use gradient accumulation, fp16 |
| Accuracy too low (<70%) | Medium | Medium | Train longer, tune hyperparameters, verify data pipeline |
| No robustness advantage | Low | Medium | Try different corruptions, focus on format generalization |
| Implementation takes too long | Medium | Low | Focus on Phases 1-3, skip Phase 4 if needed |

---

## Novelty and Contributions

### What's Novel
1. **First systematic comparison** of byte vs pixel on controlled budget
2. **Mixed-format training** for robustness
3. **Comprehensive robustness evaluation** across 4 dimensions
4. **Conservative architecture** optimized for CIFAR-10 (not just scaled-down ImageNet)
5. **Open-source reproducible implementation**

### What's Known (Building On)
- ByteFormer architecture (Apple, 2023)
- Shifted window attention (Swin Transformer)
- Byte-level modeling viability (EvaByte, BLT)

### Research Gap Filled
ByteFormer focused on ImageNet single-format; this project systematically explores:
- Format robustness and generalization
- Efficiency tradeoffs on smaller datasets
- Robustness dimensions beyond accuracy
- Hybrid approaches

---

## Documentation Index

All planning documents are complete and cross-referenced:

1. **[CLAUDE.md](CLAUDE.md)** - Project charter (scope, goals, questions)
2. **[RESEARCH.md](RESEARCH.md)** - Literature review (11 sections, 6 major works)
3. **[ENCODING_SCHEMES.md](ENCODING_SCHEMES.md)** - Approaches (baselines + byte-level)
4. **[DESIGN.md](DESIGN.md)** - Technical architecture (data, models, training)
5. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Phased plan (5 phases, 23 days)
6. **[README.md](README.md)** - Project landing page (quick start, status)
7. **[PLANNING_SUMMARY.md](PLANNING_SUMMARY.md)** - This document

---

## Next Steps for Implementation

### Immediate (Day 1)
```bash
# Phase 0: Project Setup
cd byte_level_image_encoding
uv init
uv add torch torchvision pillow numpy matplotlib tqdm pyyaml
mkdir -p src/{data,models,training,utils} configs scripts notebooks tests
python scripts/test_setup.py  # Verify CIFAR-10 loads
```

### Week 1 (Days 2-4)
- Implement pixel-level data loader
- Implement ResNet-18 baseline
- Implement training pipeline
- Train ResNet-18 for 100 epochs
- Document baseline results

### Week 2 (Days 5-10)
- Analyze CIFAR-10 byte sequence lengths
- Implement byte-level data loader (JPEG, PNG, mixed)
- Implement ByteFormer architecture
- Train 3 ByteFormer variants (JPEG, PNG, mixed)
- Compare format performance

### Week 3 (Days 11-14)
- Implement corruption utilities
- Run robustness experiments (4 types)
- Create comparison tables and plots
- Document findings

### Optional Week 4 (Days 15-21)
- Implement ViT-Tiny and hybrid model
- Run architectural ablations
- Explore optimizations

---

## Open Research Questions (Post-Implementation)

After completing planned experiments, consider:

1. **Scaling**: How does byte-level approach scale to ImageNet or larger datasets?
2. **Dense Prediction**: Can byte-level work for segmentation/detection?
3. **Generative Models**: Can byte-level models generate valid image files?
4. **Compression Learning**: Do models implicitly learn compression algorithms?
5. **Privacy**: Are there privacy advantages to processing without full decompression?
6. **Multimodal**: How effective is unified text+image byte tokenization?

---

## Confirmation for Proceeding

### Decisions Made
✅ Dataset: CIFAR-10
✅ Downsampling: 8:1 conservative
✅ Format: Mixed (JPEG + PNG)
✅ Target: Within 10% + robustness
✅ Scope: Phases 1-3 (baseline + byte + robustness)
✅ Timeline: 2-3 weeks
✅ Resources: ~103 GPU hours

### Ready to Implement
- [x] All planning documents created
- [x] Architecture fully specified
- [x] Experiments defined
- [x] Success criteria clear
- [x] Risks identified and mitigated
- [x] Phase 0 commands ready

**Status**: ✅ **READY TO PROCEED WITH IMPLEMENTATION**

---

## Summary

We have completed comprehensive planning for a byte-level image encoding research project. The plan is:

- **Ambitious but achievable**: Targets novel contributions while being computationally tractable
- **Well-grounded**: Built on recent literature (ByteFormer, EvaByte, BLT)
- **Systematic**: Controlled comparisons with clear metrics
- **Reproducible**: Fixed seeds, documented configs, phased approach
- **Flexible**: Phases can be executed independently, risks mitigated

**The project is ready to move from planning to implementation.**

---

*Planning completed: 2025-11-21*
*Ready for: Implementation Phase 0*
