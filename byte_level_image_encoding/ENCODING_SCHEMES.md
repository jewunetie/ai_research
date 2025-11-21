# Encoding Schemes: Byte-Level vs Pixel-Level

## Overview

This document defines the concrete encoding approaches we'll implement and compare for CIFAR-10 image classification.

**Design Decisions**:
- Dataset: CIFAR-10 (32×32 RGB images, 10 classes)
- Downsampling: Conservative 8:1 ratio for byte sequences
- Formats: Mixed training (JPEG quality 75 + PNG)
- Target: Within 5-10% of pixel baseline + robustness advantages

---

## Baseline: Pixel-Level Encoding

### Approach 1A: CNN Baseline (ResNet-18)
**Input**: Raw pixel arrays (32×32×3)

**Architecture**:
- Standard ResNet-18 adapted for CIFAR-10
- Input: 3×32×32 pixel tensor
- Preprocessing: Normalize to [0, 1], standard augmentations
- Output: 10-class logits

**Rationale**: Established baseline, widely used, efficient

**Expected Performance**: 92-95% accuracy on CIFAR-10

**Sequence Length**: 32×32×3 = 3,072 pixel values

---

### Approach 1B: Vision Transformer Baseline (ViT-Tiny)
**Input**: Raw pixel arrays divided into patches

**Architecture**:
- Patch size: 4×4 (gives 64 patches for 32×32 image)
- Embedding dim: 192
- Layers: 12
- Heads: 3
- Input tokens: 64 patches + 1 CLS token = 65 tokens
- Output: 10-class logits

**Rationale**: Transformer baseline for fair comparison with byte-level transformers

**Expected Performance**: 85-90% accuracy (ViT needs more data, may underperform CNN on CIFAR-10)

**Sequence Length**: 65 tokens

---

## Byte-Level Encoding Schemes

### File Size Analysis for CIFAR-10

**Raw pixels**: 32×32×3 = 3,072 bytes (uncompressed)

**Compressed formats** (typical for 32×32 image):
- PNG: 2-8 KB (2,048-8,192 bytes) - depends on complexity
- JPEG (quality 75): 1-4 KB (1,024-4,096 bytes)
- JPEG (quality 95): 2-6 KB (2,048-6,144 bytes)

**Average expected**: ~3-5 KB = 3,072-5,120 bytes

---

## Approach 2A: Raw Byte Sequence Encoding

### Input Representation
1. Load image file (JPEG or PNG) as raw bytes
2. Read entire file as byte sequence: `[b₀, b₁, b₂, ..., bₙ]` where n ≈ 3,000-5,000
3. Each byte is integer in range [0, 255]
4. Pad/truncate to fixed length: **MAX_LENGTH = 8,192 bytes**

### Architecture: ByteFormer-CIFAR (Conservative)

**Embedding**:
- Byte vocabulary: 256 tokens
- Embedding dimension: 192
- Learnable byte embedding table (256 × 192)
- Positional encoding: Learned 1D positional embeddings (8,192 × 192)

**Downsampling Stage 1**: Strided 1D Convolution
- Kernel size: 8
- Stride: 8 (conservative 8:1 ratio)
- Input: 8,192 bytes
- Output: 1,024 tokens
- Feature dimension: 192

**Transformer Blocks**:
- Architecture: Shifted Window Attention (1D)
- Window size: 128 tokens
- Number of blocks: 6
- Hidden dimension: 192
- MLP dimension: 768 (4× expansion)
- Attention heads: 3

**Hierarchical Downsampling**:
- After block 2: Downsample 1,024 → 512 (2:1)
- After block 4: Downsample 512 → 256 (2:1)
- Final sequence length: 256 tokens

**Classification Head**:
- Global average pooling over 256 tokens
- Linear projection: 192 → 10 classes

**Total Parameters**: ~8-10M (comparable to ResNet-18)

**Effective Downsampling**: 8,192 → 256 = 32:1 total ratio

---

## Approach 2B: Format-Aware Byte Encoding

### Input Representation
Parse file structure to create hierarchical representation:

**For PNG files**:
1. Parse PNG chunks (IHDR, IDAT, IEND, etc.)
2. Create chunk-level embeddings
3. Embed chunk types + chunk data separately
4. Hierarchical processing: chunk-level → byte-level

**For JPEG files**:
1. Parse JPEG segments (SOI, APP0, DQT, SOF, SOS, EOI)
2. Identify DC coefficients, AC coefficients
3. Create segment-level embeddings
4. Hierarchical processing: segment-level → byte-level

**Rationale**: Explicitly model file format structure

**Challenge**: Complex implementation, format-specific code

**Decision**: **Defer to future work** - too complex for initial experiments

---

## Approach 2C: Hybrid Byte + Pixel Encoding

### Input Representation
Process image in both modalities:

**Branch 1 (Byte-Level)**:
- ByteFormer-CIFAR processing
- Output: 256-dimensional byte representation

**Branch 2 (Pixel-Level)**:
- ResNet-18 backbone (remove final FC layer)
- Output: 512-dimensional pixel representation

**Fusion**:
- Concatenate byte + pixel features: 256 + 512 = 768
- Fusion MLP: 768 → 384 → 10 classes

**Rationale**: Combine complementary information sources

**Expected Performance**: Should exceed either modality alone

**Decision**: **Implement as Phase 4** if time permits

---

## Format-Specific Training Strategies

### Strategy A: Single Format Training
Train separate models:
- Model_JPEG: Trained only on JPEG (quality 75)
- Model_PNG: Trained only on PNG

**Evaluation**: Test each on both formats
- Cross-format accuracy: Model_JPEG on PNG, Model_PNG on JPEG

---

### Strategy B: Mixed Format Training (PRIMARY APPROACH)
Train single model on mixed data:
- 50% JPEG (quality 75)
- 50% PNG

**Data Pipeline**:
1. Load original CIFAR-10 images (PNG from torchvision)
2. On-the-fly conversion: 50% remain PNG, 50% convert to JPEG
3. Random format selection per sample per epoch

**Hypothesis**: Mixed training improves format robustness

---

### Strategy C: Format Augmentation
Treat format as augmentation:
- Same image, multiple formats
- Model sees JPEG and PNG versions in same batch

**Implementation**: Each image has probability:
- 50% JPEG (quality 75)
- 25% JPEG (quality 95)
- 25% PNG

---

## Sequence Length Comparison

| Approach | Sequence Length | Effective Length (after processing) | Memory |
|----------|----------------|-------------------------------------|--------|
| **ResNet-18 (pixels)** | 3,072 values | N/A (CNN, not sequential) | Low |
| **ViT-Tiny (patches)** | 65 tokens | 65 tokens | Low |
| **ByteFormer-CIFAR (bytes)** | 8,192 bytes | 256 tokens (after 32:1 downsample) | Medium |
| **Hybrid** | Both | 768 features (fusion) | High |

**Analysis**:
- Byte approach starts with 126× more inputs than ViT patches (8,192 vs 65)
- Conservative downsampling (8:1) → 1,024 tokens
- Hierarchical downsampling (4:1 total) → 256 final tokens
- Final byte sequence is 4× longer than ViT (256 vs 65)

---

## Robustness Evaluation Schemes

### Robustness Test 1: Byte Corruption
**Setup**: Randomly flip bits in byte sequence
- Corruption rates: 0.1%, 0.5%, 1%, 2%, 5%
- Location: Random byte positions
- Metric: Accuracy vs. corruption rate

**Hypothesis**: Byte-level models more robust to byte corruption than pixel models to pixel noise

---

### Robustness Test 2: Format Generalization
**Setup**: Train on {JPEG, PNG}, test on {WebP, BMP, TIFF}
- Convert CIFAR-10 test set to additional formats
- Zero-shot evaluation (no fine-tuning)
- Metric: Cross-format accuracy

**Baseline**: Pixel models always see same decoded pixels (format-agnostic)

**Hypothesis**: Mixed-format byte training generalizes to unseen formats

---

### Robustness Test 3: File Truncation
**Setup**: Remove trailing bytes from file
- Truncation levels: 5%, 10%, 20%, 50%
- Metric: Accuracy vs. truncation level

**Hypothesis**: Byte models learn to make predictions from partial files

---

### Robustness Test 4: Compression Artifacts
**Setup**: Test on various JPEG quality levels
- Train: JPEG quality 75
- Test: Quality levels [50, 60, 70, 80, 90, 100]
- Metric: Accuracy vs. quality level

**Baseline**: Pixel models see decoded images (artifacts in pixel space)

---

## Data Pipeline Design

### CIFAR-10 Dataset Stats
- Training: 50,000 images
- Test: 10,000 images
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Original format: 32×32 RGB PNG images

### Byte-Level Data Loader

```python
class ByteLevelCIFAR10Dataset:
    def __init__(self, split='train', format='mixed', max_length=8192):
        """
        Args:
            split: 'train' or 'test'
            format: 'jpeg', 'png', or 'mixed'
            max_length: Maximum byte sequence length
        """
        self.max_length = max_length
        self.format = format

    def __getitem__(self, idx):
        # Load PIL image
        image, label = self.cifar10[idx]

        # Convert to bytes
        if self.format == 'mixed':
            format_choice = random.choice(['jpeg', 'png'])
        else:
            format_choice = self.format

        byte_sequence = self.image_to_bytes(image, format_choice)

        # Pad/truncate
        byte_sequence = self.pad_or_truncate(byte_sequence, self.max_length)

        return torch.LongTensor(byte_sequence), label
```

### Pixel-Level Data Loader

```python
class PixelLevelCIFAR10Dataset:
    def __init__(self, split='train'):
        # Standard torchvision CIFAR-10
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
```

---

## Computational Budget Analysis

### Training Cost Estimates (per epoch on single GPU)

**ResNet-18 (Pixel)**:
- FLOPs per image: ~0.56 GFLOPs
- Total per epoch: 0.56 × 50K = 28 TFLOPs
- Time estimate: ~2 minutes (A100)

**ViT-Tiny (Pixel)**:
- FLOPs per image: ~0.3 GFLOPs (65 tokens, 12 layers)
- Total per epoch: 0.3 × 50K = 15 TFLOPs
- Time estimate: ~1 minute (A100)

**ByteFormer-CIFAR (Bytes)**:
- FLOPs per image: ~2-3 GFLOPs (1,024 tokens initial, 6 layers, window attention)
- Total per epoch: 2.5 × 50K = 125 TFLOPs
- Time estimate: ~8-10 minutes (A100)

**Computational Ratio**: ByteFormer ~4-5× slower than ResNet-18

**Training Budget**:
- 100 epochs × 10 minutes = ~17 hours for byte model
- 100 epochs × 2 minutes = ~3 hours for pixel model
- **Total**: ~20 hours for full comparison

**Verdict**: Tractable for experimentation

---

## Evaluation Metrics

### Primary Metrics
1. **Top-1 Accuracy**: Standard classification accuracy
2. **Top-5 Accuracy**: For error analysis
3. **Per-Class Accuracy**: Identify biases

### Efficiency Metrics
4. **Training Time**: Epochs × time per epoch
5. **Inference Time**: Milliseconds per image
6. **FLOPs**: Computational cost
7. **Parameters**: Model size
8. **Memory Usage**: Peak GPU memory during training

### Robustness Metrics
9. **Byte Corruption Robustness**: Accuracy @ 1% corruption
10. **Format Transfer**: Accuracy on unseen formats
11. **Truncation Robustness**: Accuracy @ 10% truncation
12. **Compression Robustness**: Accuracy across JPEG quality levels

### Format-Specific Metrics
13. **JPEG Accuracy**: Test on JPEG only
14. **PNG Accuracy**: Test on PNG only
15. **Format Gap**: |Acc_JPEG - Acc_PNG|

---

## Summary: Encoding Schemes to Implement

### Phase 1: Baseline Establishment
✅ **Approach 1A**: ResNet-18 on pixels
✅ **Approach 1B**: ViT-Tiny on patches
✅ **Approach 2A**: ByteFormer-CIFAR on bytes (mixed format)

### Phase 2: Format Analysis
✅ Train ByteFormer variants:
- JPEG-only
- PNG-only
- Mixed (primary)

### Phase 3: Robustness Evaluation
✅ All robustness tests (corruption, format transfer, truncation, compression)

### Phase 4: Advanced (Time Permitting)
⏳ **Approach 2C**: Hybrid byte + pixel model

---

## Next Steps

1. ✅ Define encoding schemes (this document)
2. ⏳ Design implementation architecture
3. ⏳ Address open questions
4. ⏳ Create IMPLEMENTATION.md with detailed plan
5. ⏳ Implement and experiment

---

*Last updated: 2025-11-21*
