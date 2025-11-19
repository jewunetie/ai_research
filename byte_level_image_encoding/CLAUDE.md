# Byte-Level Image Encoding

## Project Overview

This is an exploratory research study investigating whether models can operate directly on image file bytes (PNG/JPEG file bytes) rather than conventional pixel grids.

## Core Idea

Traditional computer vision models process images as arrays of pixels (H×W×C tensors). This project explores an alternative paradigm: treating image files as sequences of bytes and processing them directly at the byte level.

**Key Hypothesis**: By operating on file bytes rather than pixels, we might:
- Unify text and image tokenization (both as byte sequences)
- Simplify preprocessing pipelines
- Enable format-agnostic processing
- Potentially learn file format structure implicitly

## Research Goals

1. **Investigate Viability**: Can vision models be trained to operate on byte sequences from image files?
2. **Performance Comparison**: How do byte-level encodings compare to standard pixel-based encodings in terms of:
   - Classification accuracy
   - Training efficiency
   - Computational requirements
   - Robustness to file corruption
3. **Understand Tradeoffs**: What are the advantages and disadvantages of byte-level image encoding?
4. **Explore Applications**: Where might byte-level encoding be particularly beneficial?

## Technical Stack

- **Language**: Python
- **Package Manager**: uv
- **Framework**: PyTorch
- **Models**: Hugging Face Transformers (for pretrained models and architectures)
- **Datasets**: Hugging Face Datasets

## Scope

This is an **exploratory study** testing whether byte-level encoding is viable for vision tasks. The research will:
- Compare byte-level and pixel-level approaches systematically
- Document computational challenges and practical constraints
- Identify scenarios where byte-level encoding might be advantageous
- Provide concrete experimental results on standard benchmarks

## Research Questions

1. Has byte-level image encoding been explored in prior work?
2. What is the optimal architecture for processing byte sequences from images?
3. How does sequence length compare between byte sequences and pixel patches?
4. Can models learn to ignore file format noise and focus on visual content?
5. What are the computational bottlenecks?

## Status

- [x] Project initialized
- [ ] Literature review completed
- [ ] Research summary documented
- [ ] Encoding schemes defined
- [ ] Implementation plan created
- [ ] Experiments conducted
- [ ] Results analyzed
