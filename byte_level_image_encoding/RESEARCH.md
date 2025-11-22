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

### 6.1 **🔥 NOVEL: Format-Agnostic Zero-Shot Transfer (UNTESTED)**

**The Core Question**: Can a model trained **only** on JPEG bytes classify PNG/WebP/BMP images with **zero fine-tuning**?

#### Why This Is Novel

**Existing Work Limitations**:
- **ByteFormer** (Apple 2023): Reports separate results for TIFF (77.3%), PNG (77%+), and JPEG (65.92%), but **does NOT test cross-format transfer**
- **EvaByte** (2025): Uses JPEG for images, but **no evaluation on other formats**
- **BLT** (Meta 2024): Focuses on text, minimal vision experiments
- **No published work** has systematically studied: *"Train on format A, test on format B with zero shots"*

#### Evidence This Has NOT Been Done

From web research conducted 2025-11-22:

1. **ByteFormer Paper Analysis**:
   - GitHub repo contains separate training configurations for TIFF, PNG, and JPEG
   - Each format trained and tested independently
   - **No cross-format evaluation reported in paper or code**

2. **Community Reports**:
   - Stack Overflow user reported **20% accuracy drop** when training CNN on JPEGs, testing on PNGs (empirical evidence of format shift)
   - General recommendation: "If you want to classify both PNG and JPG, train on both"
   - **This suggests cross-format transfer is a known problem, but unstudied in byte-level models**

3. **Cross-Modal Transfer (bGPT)**:
   - bGPT (Microsoft, Feb 2024) showed **positive transfer** between images and audio bytes
   - **Negative transfer** between text and images (different byte patterns)
   - **Key insight**: "Shared byte patterns exist between modalities like audio and images"
   - **Implication**: Cross-format transfer might work if formats share byte-level patterns

4. **Domain Shift Research**:
   - Extensive literature on distribution shift (camera changes, compression quality)
   - **NO research** on file format as a domain shift factor in byte-level models
   - Compression robustness studied (JPEG quality factors), but not JPEG→PNG transfer

5. **ByteNet (Oct 2024)**:
   - Uses byte-to-image visualization for file fragment classification
   - Focuses on distinguishing formats (JPEG vs PNG vs Audio)
   - **Orthogonal goal**: Classify file type, not transfer across types

#### Why This Question Matters

**Scenario A: High Transfer (PNG acc ≈ JPEG acc)**
- **Implication**: Byte models learn **content-agnostic representations**
- **Contribution**: First demonstration that byte-level features transfer across radically different encodings
- **Impact**: Suggests byte models are more powerful than previously thought
- **Applications**: Train once on cheap format (JPEG), deploy on any format

**Scenario B: Zero Transfer (PNG acc ≈ random)**
- **Implication**: Byte models are **format-specific**, overfitted to encoding schemes
- **Contribution**: Fundamental limitation of byte-level approaches
- **Impact**: Hybrid pixel-byte models needed for format robustness
- **Applications**: Must train on all formats users will encounter

**Scenario C: Partial Transfer (PNG acc = 40-60%)**
- **Implication**: Some byte patterns transfer, others don't
- **Contribution**: Identify which patterns are content vs. format specific
- **Impact**: Design better architectures that separate format and content
- **Applications**: Format adaptation layers, domain adaptation techniques

#### Technical Insight: Why Formats Are Radically Different

Same cat photo encoded as JPEG vs PNG uses **completely different byte sequences**:

| Aspect | JPEG (Quality 75) | PNG (Lossless) |
|--------|------------------|----------------|
| Compression | DCT + Huffman + Quantization | Deflate (LZSS + Huffman) |
| Color Space | YCbCr (luminance + chroma) | RGB channels |
| Structure | Progressive scans, restart markers | Scanline filtering, IDAT chunks |
| Byte Patterns | Non-aligned Huffman codes | Byte-aligned chunk structure |
| Headers | JFIF/EXIF metadata | PNG signature + critical chunks |
| File Size (32×32 CIFAR) | ~1-3 KB | ~3-8 KB |

**Key Point**: These are as different as English and French text, yet encode identical visual semantics.

#### Minimal Compute Experiment Design

```python
# Total compute: ~20 GPU hours on RTX 3090

# 1. Encode CIFAR-10 in 4 formats (1 hour)
formats = ['jpeg_q75', 'png', 'webp', 'bmp']
for image in cifar10:
    save_as_jpeg(image, quality=75)
    save_as_png(image)
    save_as_webp(image)
    save_as_bmp(image)

# 2. Train ONLY on JPEG (12-15 hours)
model = ByteFormer(max_bytes=8192, embed_dim=192, depth=6)
model.train(cifar10_jpeg_train)  # 50K images, JPEG only

# 3. Zero-shot test on all formats (< 1 hour)
results = {
    'jpeg': model.test(cifar10_jpeg_test),   # ~80-85% expected (in-distribution)
    'png': model.test(cifar10_png_test),     # ??? (NOVEL)
    'webp': model.test(cifar10_webp_test),   # ??? (NOVEL)
    'bmp': model.test(cifar10_bmp_test),     # ??? (NOVEL - uncompressed baseline)
}

# 4. Analysis
- Plot accuracy vs format
- Analyze attention patterns: Do they align to semantic content or format structure?
- Failure analysis: Which classes transfer, which don't?
```

**Expected Results (All Publishable)**:
- **High transfer**: CVPR/ICCV paper (major finding)
- **Zero transfer**: NeurIPS paper (important negative result)
- **Partial transfer**: Both conferences (analysis of what transfers)

#### Related Findings from Literature

**Compression Robustness (Different from Format Transfer)**:
- CNNs robust to JPEG quality ≥ 10 (Ehrlich et al., ICCV 2021)
- Fine-tuning with JPEG augmentation mitigates compression artifacts
- **But**: This studies JPEG-Q50 vs JPEG-Q95, not JPEG vs PNG

**Cross-Modal Transfer (bGPT)**:
- Positive transfer: ImageNet bytes → Audio tasks (+5-10% vs random init)
- Negative transfer: Text bytes → Audio/Images (text has distinct patterns)
- **Implication for formats**: If image→audio transfers, maybe JPEG→PNG could too

**Domain Shift**:
- Camera changes, lighting, background cause distribution shift
- No prior work treats file format as a domain
- **Gap**: Format shift is a type of distribution shift, unstudied in byte models

#### Novelty Summary

✅ **Never tested**: No published work on cross-format zero-shot transfer in byte models
✅ **Clear hypothesis**: Format-agnostic vs format-specific representations
✅ **Minimal compute**: 20 GPU hours on CIFAR-10
✅ **All outcomes interesting**: Success OR failure is publishable
✅ **Fast iteration**: Results in 2-3 days
✅ **Addresses fundamental question**: Do byte models learn content or format?

**Recommendation**: **Prioritize this experiment** as the primary novel contribution of this research.

---

### 6.2 Systematic Comparisons Needed
- **Same dataset, same budget**: Pixel vs. byte-level models
- **Multiple formats**: Train on one, test on others (generalization) ← **UPDATED: See 6.1 above**
- **Multiple tasks**: Classification, segmentation, detection
- **Multiple scales**: CIFAR (small) to ImageNet (large)

### 6.3 Architectural Exploration
- **State space models** (Mamba, S4): Better for long sequences?
- **Hybrid models**: Combine byte-level and pixel-level representations
- **Hierarchical processing**: Parse file structure explicitly
- **Learned compression**: Can models learn to compress byte sequences optimally?

### 6.4 Format Robustness
- **Mixed-format training**: Does it improve generalization?
- **Format conversion**: Byte-level models for transcoding?
- **Corruption robustness**: Random byte flips, truncation, format errors
- **Zero-shot format transfer**: See Section 6.1 for primary novel direction

### 6.5 Efficiency Improvements
- **Better downsampling**: Content-aware vs. fixed-stride
- **Sparse attention**: Which byte positions matter most?
- **Knowledge distillation**: Train large byte model, distill to smaller pixel model

### 6.6 Practical Applications
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

#### **Hypothesis 1: Format-Agnostic Zero-Shot Transfer (PRIMARY NOVEL HYPOTHESIS)**
**H1**: A byte-level model trained on JPEG will achieve **>50% of its in-distribution accuracy** when tested zero-shot on PNG/WebP/BMP formats.

**Experiment** (Detailed in Section 6.1):
- Train ByteFormer-CIFAR on JPEG-only (50K images)
- Test zero-shot on PNG, WebP, BMP (10K images each)
- Baseline: Pixel-level CNN for comparison
- Metric: Accuracy ratio (out-of-format / in-format)

**Success Criteria**:
- **Strong transfer**: PNG accuracy ≥ 70% of JPEG accuracy → Major positive result
- **Partial transfer**: PNG accuracy = 40-70% of JPEG → Analyze what transfers
- **Zero transfer**: PNG accuracy ≤ 30% of JPEG → Important negative result (format-specific learning)

**Compute**: 20 GPU hours total
**Timeline**: 3-4 days

---

#### Hypothesis 2: Mixed-Format Training Improves Generalization
**H2**: Models trained on mixed formats (JPEG + PNG) will generalize better to unseen formats (WebP, BMP) than single-format training.

**Experiment**:
- Train 3 models: JPEG-only, PNG-only, Mixed (50% JPEG + 50% PNG)
- Test on WebP and BMP (unseen formats)
- Metric: Zero-shot accuracy on WebP/BMP

**Prediction**: Mixed > JPEG-only and Mixed > PNG-only

---

#### Hypothesis 3: Robustness to Byte Corruption
**H3**: Byte-level models will be more robust to file corruption (random byte flips) than pixel-level models are to pixel corruption.

**Experiment**:
- Compare byte-model (byte corruption) vs. pixel-model (pixel corruption)
- Vary corruption rate: 0.1%, 1%, 5%, 10%
- Metric: Accuracy degradation curve

---

#### Hypothesis 4: Efficient Downsampling
**H4**: Content-aware downsampling (entropy-based) will outperform fixed-stride downsampling for byte sequences.

**Experiment**:
- Compare fixed-stride vs. entropy-based patching (inspired by BLT)
- Control for compute budget
- Metric: Accuracy per FLOP

---

#### Hypothesis 5: Hybrid Models
**H5**: Models using both byte-level and pixel-level representations will outperform either alone.

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
1. ❓ **Do byte models learn format-agnostic or format-specific representations?** ← **PRIMARY NOVEL QUESTION**
2. ❓ Can models transfer zero-shot across formats (JPEG → PNG)?
3. ❓ How do byte-level models perform on dense prediction (segmentation, detection)?
4. ❓ Can mixed-format training improve robustness?
5. ❓ What's the optimal downsampling strategy?
6. ❓ How do hybrid (byte + pixel) models compare?
7. ❓ Can byte-level models match pixel-level efficiency?

### Research Opportunity
**PRIMARY CONTRIBUTION: Format-Agnostic Zero-Shot Transfer**
- **Never tested**: No prior work on cross-format zero-shot transfer
- **Minimal compute**: 20 GPU hours on CIFAR-10
- **All outcomes publishable**: Success OR failure is a significant finding
- **Addresses fundamental question**: Content vs. format learning

**SECONDARY: Systematic comparison of byte-level vs. pixel-level encodings** across:
- Multiple datasets (CIFAR, ImageNet)
- Multiple tasks (classification, maybe reconstruction)
- Multiple formats (JPEG, PNG, WebP, BMP)
- Controlled compute budgets

This project primarily addresses question 1-2 (novel), with secondary focus on questions 3-7.

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
1. **🔥 PRIMARY NOVEL CONTRIBUTION: Format-Agnostic Zero-Shot Transfer**
   - **First systematic study** of cross-format transfer in byte-level models
   - **Untested question**: Do byte models learn content or format?
   - **All outcomes publishable**: High/partial/zero transfer all significant findings
   - **Minimal compute**: Feasible with 20 GPU hours on CIFAR-10

2. **SECONDARY CONTRIBUTIONS**:
   - **Systematic comparison**: Controlled experiments (same dataset, budget)
   - **Multiple formats**: Test JPEG, PNG, WebP, BMP comprehensively
   - **Robustness analysis**: Byte corruption vs. pixel corruption
   - **Hybrid approaches**: Combine byte and pixel representations
   - **Clear documentation**: Open research process, reproducible results

### Why It Matters
- **ByteFormer** (2023): Trained separate models per format, **no cross-format testing**
- **EvaByte** (2025): Uses JPEG only, **no format generalization study**
- **Community evidence**: 20% accuracy drop reported when training on JPEG, testing on PNG (uncontrolled)
- **bGPT** (2024): Showed cross-modal transfer (images→audio), suggests cross-format might work
- **Domain shift literature**: Extensive work on camera/compression changes, **but not format as domain**
- **ByteNet** (2024): Classifies file types, doesn't test transfer across types

**Gap**: No published work has asked "Can a byte model trained on JPEG classify PNG images?"

**This project addresses this fundamental question with minimal compute.**

---

## References

### Core Byte-Level Models
1. Buch, S., et al. (2023). Bytes Are All You Need: Transformers Operating Directly On File Bytes. arXiv:2306.00238
2. EvaByte Team (2025). EvaByte: Efficient Byte-level Language Models at Scale. https://hkunlp.github.io/blog/2025/evabyte/
3. Meta AI (2024). Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871
4. Zhang, Y., et al. (2025). Unified Multimodal Understanding via Byte-Pair Visual Encoding. arXiv:2506.23639
5. Jaegle, A., et al. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. arXiv:2107.14795
6. Xue, L., et al. (2021). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. arXiv:2105.13626

### Cross-Modal Transfer and Domain Shift
7. Microsoft Research (Feb 2024). Beyond Language Models: Byte Models are Digital World Simulators (bGPT). arXiv:2402.19155
8. ByteNet Team (Oct 2024). ByteNet: Rethinking Multimedia File Fragment Classification through Visual Perspectives. arXiv:2410.20855
9. Hendrycks, D., et al. (2020). Measuring Robustness to Natural Distribution Shifts in Image Classification. NeurIPS 2020
10. Ehrlich, M., et al. (2021). Analyzing and Mitigating JPEG Compression Defects in Deep Learning. ICCV 2021 Workshop

### Related Work
11. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). ICLR 2021
12. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021
13. Ballé, J., et al. (2018). Variational Image Compression with a Scale Hyperprior. ICLR 2018

### Community Evidence
14. Stack Overflow (2017). "Does the image format (png, jpg, gif) affect how an image recognition neural net is trained?" https://stats.stackexchange.com/questions/285931
15. Stack Overflow (2017). "Will jpeg compression affect training and classification using Convolutional Neural Networks?" https://stackoverflow.com/questions/47497352

---

*Last updated: 2025-11-22 (Added format-agnostic zero-shot transfer research)*
