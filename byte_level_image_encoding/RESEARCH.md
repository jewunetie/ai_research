# Research Summary: Byte-Level Image Encoding

## Executive Summary

Byte-level image encoding is an **emerging research area** (2023-2025) where models process image file bytes directly rather than decoded pixel arrays. Several significant works demonstrate viability, but systematic comparisons and further exploration remain valuable research directions.

**Key Finding**: Byte-level image models have been successfully demonstrated, particularly ByteFormer (Apple, 2023) achieving 77.3% ImageNet accuracy. However, challenges around sequence length and computational efficiency persist.

---

## 1. Prior Work on Byte-Level Image Models

### 1.1 ByteFormer (Apple, 2023) ⭐ **Most Directly Relevant**

**Paper**: "Bytes Are All You Need: Transformers Operating Directly On File Bytes"
**Authors**: Apple Machine Learning Research
**Code**: https://github.com/apple/corenet/tree/main/projects/byteformer

#### Key Contributions
- **Direct file byte processing**: Operates on PNG, JPEG, TIFF without decoding to pixels
- **ImageNet Top-1 accuracy**: 77.33% (TIFF format)
- **Parameter efficiency**: 8.8M parameters vs. Perceiver IO's 62.3M at comparable accuracy
- **Modality agnostic**: Same architecture achieves 95.51% on Speech Commands V2 (audio)
- **Multimodal capability**: Single model trains on ImageNet + audio with minimal degradation

#### Architecture Details
- **Byte embedding**: 256 possible byte values → d-dimensional vectors
- **Downsampling**: Strided 1D convolution (kernel size 32, stride 16)
- **Attention**: Shifted window attention (window size 128) instead of O(n²) full attention
- **Hierarchical downsampling**: After blocks 0, 1, 3, 5, 7, 9

#### Sequence Length Analysis
Input format determines sequence length:
- **fHWC/TIFF**: ~9,407-9,415 tokens (after downsampling)
- **PNG**: ~9,428 tokens
- **JPEG (quality 100)**: ~12,140 tokens
- **Baseline ViT**: 196 patch tokens (48-62× shorter!)

Raw 224×224 TIFF: ~150,528 bytes before downsampling

#### Format Sensitivity
- **TIFF/PNG**: 77%+ accuracy
- **JPEG**: 65.92% accuracy (Huffman encoding complexity)
- **Random shuffling**: Drops JPEG accuracy to 3.06% (locality is critical)

#### Computational Cost
- **23.74 billion FLOPs** on A100 GPU
- No file decoding at inference (only byte embedding + transformer)

---

### 1.2 EvaByte (HKU + SambaNova, January 2025)

**Website**: https://hkunlp.github.io/blog/2025/evabyte/
**Code**: https://github.com/OpenEvaByte/evabyte
**Model**: 6.5B parameters

#### Key Contributions
- **First open-source byte-level model** matching tokenizer-based LM performance
- **Vision-language integration**: Seamlessly interleaves image (JPEG) with text bytes
- **Training**: 1.5T bytes of text, math, code
- **Efficiency**: Trained with 5× less data than comparable models, 2× faster decoding

#### Vision Capabilities
After 20K fine-tuning steps on ~3M images:
- Zero-shot caption generation
- Basic visual question answering
- Image generation (though "a bit fuzzily")

**Technical Approach**: Uses JPEG format to keep byte sequence length manageable

---

### 1.3 Byte Latent Transformer (Meta AI, December 2024)

**Paper**: "Byte Latent Transformer: Patches Scale Better Than Tokens" (arXiv:2412.09871)

#### Key Contributions
- **Tokenizer-free architecture** processing raw bytes
- **Dynamic patching**: Variable-sized byte groupings based on data complexity
- **Scaling efficiency**: 50% fewer FLOPs at inference vs. Llama 3 with matched performance

#### Patching Strategies
1. **Strided**: Fixed-size byte groups (e.g., every 4 bytes)
2. **Space**: Boundaries at whitespace/linguistic units
3. **Entropy**: Segments where next-byte prediction uncertainty is high (most sophisticated)

#### Architecture
- **Local encoder/decoder**: Lightweight transformers for byte-level ops
- **Global transformer**: Large model processing patch representations
- **Hash n-gram embeddings** + cross-attention between byte/patch levels

#### Performance
- **Character-level tasks**: 99.9% accuracy on spelling manipulation (vs. near-zero for Llama 3)
- **Robustness**: Superior on noise and multilingual tasks
- **Implication for vision**: Byte-level processing can extend to any sequential data

---

### 1.4 PixelBytes (October 2024)

**Paper**: "PixelBytes: Catching Unified Representation for Multimodal Generation" (arXiv:2410.01820)

#### Key Contributions
- **Unified representation**: Integrates text, audio, action-state, pixelated images (sprites)
- **PxByEmbed technique**: Creates unified space for pixel and byte data
- **Architectures explored**: RNNs, SSMs, Attention-based, LSTMs

#### Findings
- Autoregressive models outperform predictive models
- Diffusion models applicable to control problems with parallelized generation

**Note**: Focused on multimodal generation; less directly relevant to standard vision tasks

---

### 1.5 Unified Multimodal Understanding via Byte-Pair Visual Encoding (June 2025)

**Paper**: arXiv:2506.23639 (Zhang et al.)
**Publication**: ICCV 2025

#### Key Contributions
- Applies **byte-pair encoding** (BPE) to visual tokens
- **Priority-guided encoding**: Considers frequency + spatial consistency
- **Multi-stage training**: Curriculum-driven data composition
- Improved performance across diverse vision-language tasks

**Approach**: Treats visual tokens similarly to text tokens, incorporating structural information directly

---

### 1.6 Perceiver IO (DeepMind, 2021)

**Paper**: "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (arXiv:2107.14795)

#### Key Contributions
- **Modality-agnostic architecture**: Works on text, images, audio, video, point clouds
- **Byte-level language processing**: Directly processes UTF-8 bytes
- **Latent space computation**: Bulk compute in smaller latent space (O(n) vs O(n²))

#### Performance
- Byte-level Perceiver IO matches BERT (SentencePiece tokens) at same FLOPs
- ImageNet: 72.70% accuracy with 62.3M parameters

**Limitation**: Higher parameter count than ByteFormer for comparable accuracy

---

## 2. Sequence Length Comparison

### Pixel-Based Encoding (Baseline)
- **Raw pixels**: H × W × C = 224 × 224 × 3 = 150,528 values
- **ViT patches**: 224/16 × 224/16 = 196 tokens (16×16 patches)
- **Typical range**: 196-784 tokens depending on patch size

### Byte-Based Encoding
| Format | File Size (approx) | Tokens (after downsampling) | Raw Bytes |
|--------|-------------------|---------------------------|-----------|
| TIFF (uncompressed) | ~150 KB | ~9,407 | ~150,528 |
| PNG (lossless) | ~50-150 KB | ~9,428 | ~51,200-153,600 |
| JPEG (quality 100) | ~20-60 KB | ~12,140 | ~20,480-61,440 |
| JPEG (quality 75) | ~5-20 KB | ~3,000-8,000 | ~5,120-20,480 |

### Key Observations
1. **Byte sequences are 48-62× longer** than ViT patch tokens (before downsampling)
2. **Compression helps**: JPEG reduces sequence length but hurts accuracy (ByteFormer: 65.92% vs 77.33%)
3. **Downsampling is critical**: ByteFormer uses 32:16 strided convolution to reduce sequence length
4. **For text**: Byte sequences are 3.8× longer than tokenized sequences

---

## 3. Computational Challenges

### Challenge 1: Sequence Length
**Problem**: Transformer attention is O(n²) in sequence length
**Solutions**:
- **Shifted window attention** (ByteFormer): Local attention within windows
- **Hierarchical downsampling**: Progressive reduction of sequence length
- **Latent space processing** (Perceiver IO): Compress to smaller latent representation
- **Dynamic patching** (BLT): Adaptive grouping based on entropy

### Challenge 2: Information Density
**Problem**: Bytes encode format structure, not just visual content
**Observations**:
- Models must learn to ignore file headers, chunk metadata, compression artifacts
- JPEG's Huffman encoding is harder to process than raw bytes (TIFF/PNG)
- Byte ordering matters: random shuffling catastrophically degrades performance

### Challenge 3: Format Dependencies
**Problem**: Different formats → different byte sequences for same image
**Findings**:
- ByteFormer accuracy varies by format (TIFF: 77%, JPEG: 66%)
- Training on mixed formats could improve robustness
- Format-specific biases may emerge

### Challenge 4: Computational Cost
**Comparison** (ImageNet, similar accuracy):
- **ViT-Ti**: 1.3B FLOPs (on 196 tokens)
- **ByteFormer**: 23.74B FLOPs (on ~9,400 tokens)
- **Cost ratio**: ~18× more compute for byte-level approach

**Mitigation**: Parameter efficiency (8.8M vs 62M for Perceiver IO) partially compensates

---

## 4. Advantages of Byte-Level Encoding

### 4.1 Unified Multimodal Representation
- **Text + images**: Both are byte sequences (EvaByte demonstration)
- **Simplified pipelines**: No modality-specific preprocessing
- **Architectural unification**: Same model architecture across modalities

### 4.2 Format Agnosticism (Potential)
- Could process any image format without format-specific decoders
- Reduces technical debt from maintaining multiple preprocessing pipelines
- Currently limited by format-specific performance differences

### 4.3 Robustness to Pixel-Level Perturbations
- Adversarial attacks targeting pixel space may be less effective
- Compressed representations (JPEG) provide implicit robustness
- **Note**: This advantage is theoretical; not empirically validated in literature

### 4.4 Learning File Structure
- Models may implicitly learn compression algorithms
- Potential for format conversion, error correction
- Could understand relationships between file size and content

### 4.5 Privacy Considerations
- Processing without full decompression may preserve privacy in some scenarios
- Selective processing of file regions possible

---

## 5. Disadvantages and Open Questions

### 5.1 Computational Inefficiency
- 18-50× more FLOPs than pixel-based approaches
- Longer training times
- Higher memory requirements during training

### 5.2 Format Sensitivity
- Performance varies significantly by format (TIFF: 77%, JPEG: 66%)
- Optimal format unclear (compression vs. processing difficulty tradeoff)
- Mixed-format training not well explored

### 5.3 Limited Benchmarking
- **ImageNet**: Primary benchmark (ByteFormer)
- **Missing**: CIFAR-10/100, COCO, ADE20K, fine-grained classification
- **No studies**: Segmentation, detection, dense prediction at byte level

### 5.4 Architecture Constraints
- Shifted window attention may miss long-range dependencies
- Downsampling choices arbitrary (32:16 in ByteFormer)
- Limited exploration of alternatives (hierarchical transformers, state space models)

### 5.5 Interpretability
- What do models learn from byte sequences?
- How much is format structure vs. visual content?
- Can we visualize byte-level attention patterns meaningfully?

---

## 6. Research Gaps and Opportunities

### 6.1 Systematic Comparisons Needed
- **Same dataset, same budget**: Pixel vs. byte-level models
- **Multiple formats**: Train on one, test on others (generalization)
- **Multiple tasks**: Classification, segmentation, detection
- **Multiple scales**: CIFAR (small) to ImageNet (large)

### 6.2 Architectural Exploration
- **State space models** (Mamba, S4): Better for long sequences?
- **Hybrid models**: Combine byte-level and pixel-level representations
- **Hierarchical processing**: Parse file structure explicitly
- **Learned compression**: Can models learn to compress byte sequences optimally?

### 6.3 Format Robustness
- **Mixed-format training**: Does it improve generalization?
- **Format conversion**: Byte-level models for transcoding?
- **Corruption robustness**: Random byte flips, truncation, format errors

### 6.4 Efficiency Improvements
- **Better downsampling**: Content-aware vs. fixed-stride
- **Sparse attention**: Which byte positions matter most?
- **Knowledge distillation**: Train large byte model, distill to smaller pixel model

### 6.5 Practical Applications
- **Unified multimodal models**: Text + images in single byte vocabulary
- **Privacy-preserving inference**: Process without full decompression
- **Compression learning**: Models that understand and generate compressed formats

---

## 7. Feasibility Assessment for This Project

### 7.1 Has Byte-Level Image Encoding Been Tried?
**Yes**, convincingly demonstrated by:
- ByteFormer (2023): 77.3% ImageNet, 8.8M params
- EvaByte (2025): Vision-language with byte-level JPEG

### 7.2 Main Bottlenecks
1. **Sequence length**: 48-62× longer than patch tokens
2. **Compute**: 18× more FLOPs than ViT
3. **Format complexity**: JPEG harder than TIFF/PNG
4. **Architecture tuning**: Shifted window attention required

### 7.3 Can We Make This Tractable?
**Yes**, with constraints:
- **Small images**: CIFAR-10 (32×32) → ~3-10 KB files → ~3,000-10,000 bytes
- **Efficient architectures**: Shifted window attention, hierarchical downsampling
- **Compressed formats**: JPEG (quality 75) → smaller files
- **Modern hardware**: A100 GPUs can handle sequences up to 10-20K tokens

### 7.4 Concrete Testable Hypotheses

#### Hypothesis 1: Format Generalization
**H1**: Models trained on mixed formats (JPEG + PNG) will generalize better to unseen formats than single-format training.

**Experiment**:
- Train 3 models: JPEG-only, PNG-only, Mixed
- Test on JPEG, PNG, WebP, BMP
- Metric: Accuracy and cross-format transfer

#### Hypothesis 2: Robustness to Byte Corruption
**H2**: Byte-level models will be more robust to file corruption (random byte flips) than pixel-level models are to pixel corruption.

**Experiment**:
- Compare byte-model (byte corruption) vs. pixel-model (pixel corruption)
- Vary corruption rate: 0.1%, 1%, 5%, 10%
- Metric: Accuracy degradation curve

#### Hypothesis 3: Efficient Downsampling
**H3**: Content-aware downsampling (entropy-based) will outperform fixed-stride downsampling for byte sequences.

**Experiment**:
- Compare fixed-stride vs. entropy-based patching
- Control for compute budget
- Metric: Accuracy per FLOP

#### Hypothesis 4: Hybrid Models
**H4**: Models using both byte-level and pixel-level representations will outperform either alone.

**Experiment**:
- Baseline: Pixel-only, Byte-only
- Hybrid: Concatenate/fuse byte and pixel representations
- Metric: Accuracy, efficiency, robustness

---

## 8. Related Work: Image Compression and Neural Codecs

### Neural Image Compression
- **Learned compression**: Models like Ballé et al. (2018) learn to compress images
- **Connection**: Byte-level models implicitly learn to process compressed representations
- **Opportunity**: Could byte-level models learn to generate compressed formats?

### Vision Transformers
- **ViT** (Dosovitskiy et al., 2020): 16×16 patches → 196 tokens
- **Swin Transformer**: Hierarchical shifted windows (similar to ByteFormer)
- **Connection**: ByteFormer adapts shifted windows from Swin to 1D byte sequences

### Multimodal Models
- **CLIP**: Contrastive text-image pretraining
- **Flamingo**: Vision-language with interleaved inputs
- **Connection**: Byte-level encoding could unify text and image vocabularies

---

## 9. Key Takeaways

### What We Know
1. ✅ Byte-level image models work (77% ImageNet)
2. ✅ Specialized architectures (shifted windows) handle long sequences
3. ✅ Parameter efficiency possible (8.8M vs. 62M)
4. ✅ Multimodal unification feasible (EvaByte)
5. ✅ Format matters (TIFF > JPEG for accuracy)

### What We Don't Know
1. ❓ How do byte-level models perform on dense prediction (segmentation, detection)?
2. ❓ Can mixed-format training improve robustness?
3. ❓ What's the optimal downsampling strategy?
4. ❓ How do hybrid (byte + pixel) models compare?
5. ❓ Can byte-level models match pixel-level efficiency?

### Research Opportunity
**Systematic comparison of byte-level vs. pixel-level encodings** across:
- Multiple datasets (CIFAR, ImageNet)
- Multiple tasks (classification, maybe reconstruction)
- Multiple formats (JPEG, PNG)
- Controlled compute budgets

This project addresses gaps 1-4 above.

---

## 10. Recommended Reading

### Core Papers (Must Read)
1. **ByteFormer**: Buch et al. (2023) - "Bytes Are All You Need: Transformers Operating Directly On File Bytes"
2. **Byte Latent Transformer**: Meta AI (2024) - "Byte Latent Transformer: Patches Scale Better Than Tokens"
3. **EvaByte**: HKU + SambaNova (2025) - Blog post on efficient byte-level models

### Background (Helpful Context)
4. **ByT5**: Xue et al. (2021) - "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
5. **Perceiver IO**: Jaegle et al. (2021) - "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
6. **Swin Transformer**: Liu et al. (2021) - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"

### Datasets
- **ImageNet**: 1.28M training images, 1000 classes
- **CIFAR-10**: 60K images (32×32), 10 classes
- **CIFAR-100**: 60K images (32×32), 100 classes

---

## 11. Novelty of This Project

### What's New Here
1. **Systematic comparison**: Controlled experiments (same dataset, budget)
2. **Multiple formats**: Test JPEG, PNG, maybe WebP
3. **Robustness analysis**: Byte corruption vs. pixel corruption
4. **Hybrid approaches**: Combine byte and pixel representations
5. **Clear documentation**: Open research process, reproducible results

### Why It Matters
- ByteFormer focused on single-format ImageNet
- Limited exploration of format generalization
- No systematic efficiency analysis
- Hybrid approaches unexplored
- Recent (2023-2025) means much is still unknown

**This project fills gaps while building on strong foundations.**

---

## References

1. Buch, S., et al. (2023). Bytes Are All You Need: Transformers Operating Directly On File Bytes. arXiv:2306.00238
2. EvaByte Team (2025). EvaByte: Efficient Byte-level Language Models at Scale. https://hkunlp.github.io/blog/2025/evabyte/
3. Meta AI (2024). Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871
4. Zhang, Y., et al. (2025). Unified Multimodal Understanding via Byte-Pair Visual Encoding. arXiv:2506.23639
5. Jaegle, A., et al. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. arXiv:2107.14795
6. Xue, L., et al. (2021). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. arXiv:2105.13626

---

*Last updated: 2025-11-19*
