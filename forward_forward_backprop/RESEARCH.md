# Research Summary: Forward-Forward with Backprop

## Table of Contents
1. [The Forward-Forward Algorithm](#the-forward-forward-algorithm)
2. [Existing Implementations](#existing-implementations)
3. [Hybrid Approaches](#hybrid-approaches)
4. [Performance Analysis](#performance-analysis)
5. [Common Experimental Setups](#common-experimental-setups)
6. [Our Contribution](#our-contribution)

---

## The Forward-Forward Algorithm

### Overview
The Forward-Forward (FF) algorithm was introduced by Geoffrey Hinton in December 2022 at NeurIPS as a novel alternative to backpropagation.

**Key Paper**: Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345

### Core Concept
FF replaces the forward and backward passes of backpropagation with **two forward passes**:
1. **Positive Pass**: Process real (positive) data
2. **Negative Pass**: Process negative data (incorrect labels or corrupted samples)

### Key Characteristics

**Local Learning**:
- Each layer has its own objective function
- No gradient backpropagation through the network
- Layer objectives: high "goodness" for positive data, low "goodness" for negative data

**Goodness Function**:
- Original formulation: Sum of squared neuron activities
- Formula: `goodness(h) = Σ(h_i²)` where h is the layer activation
- Alternative: Negative sum of squared activities
- Layer normalization required to prevent trivial solutions

**Biological Plausibility**:
- Addresses weight transport problem (backprop requires symmetric weights)
- No need for bidirectional synaptic communication
- Compatible with unidirectional neural action potentials
- Enables layer-wise, local credit assignment

### Label Embedding for Supervised Learning
For MNIST-style tasks:
- **Positive samples**: `merge(x, y)` - image with correct label embedded
- **Negative samples**: `merge(x, y_random)` - image with incorrect label
- Common implementation: Use top-left pixels (always zero in MNIST) for one-hot label encoding
- Alternative: Learnable linear projection for label encoding

### Advantages
1. **Biological plausibility**: More aligned with neuroscience findings
2. **Black-box compatibility**: Works with non-differentiable modules
3. **Temporal separation**: Positive and negative passes can be done at different times
4. **Memory efficiency**: No need to store intermediate activations for backward pass
5. **Hardware efficiency**: Potential for low-power implementations

### Limitations
1. **Performance gap**: Generally underperforms standard backpropagation
2. **Negative sample generation**: Task-dependent and can be challenging
3. **Architecture constraints**: Difficult to apply to deep CNNs effectively
4. **Label embedding requirement**: Input data must be modified to include labels
5. **Convergence speed**: Slower than backpropagation (3-5x more epochs)

---

## Existing Implementations

### Official and Community Implementations

1. **PyTorch Examples** (Official)
   - Repository: pytorch/examples/mnist_forward_forward
   - Status: Maintained
   - Features: MNIST baseline implementation

2. **mpezeshki/pytorch_forward_forward**
   - One of the first community implementations
   - Clean PyTorch code
   - MNIST and simple datasets

3. **carloalbertobarbano/forward-forward-pytorch**
   - Hard negatives sampling strategy
   - Enhanced negative sample generation

4. **loeweX/Forward-Forward**
   - Comprehensive reimplementation
   - Multiple dataset support

5. **Keras Implementation**
   - Official Keras example
   - TensorFlow backend
   - Tutorial-style documentation

### Implementation Patterns
Common across implementations:
- Fully connected networks (2-4 hidden layers)
- Layer sizes: 500-2000 neurons per layer
- ReLU activations
- Threshold-based goodness optimization
- Adam optimizer

---

## Hybrid Approaches

### Existing Hybrid Research

#### 1. Integrated Forward-Forward Algorithm (2023)
**Paper**: arXiv:2305.12960 - "The Integrated Forward-Forward Algorithm: Integrating Forward-Forward and Shallow Backpropagation With Local Losses"

**Approach**:
- Combines FF with shallow backpropagation
- Uses local losses throughout
- Maintains biological plausibility

**Results**:
- Outperformed pure FF on MNIST
- Superior noise resilience compared to standard backpropagation
- Still uses local learning rules

**Limitation**: Shallow backprop only, not full global optimization

#### 2. Deep-CBN: CNN + BiFormer + Hybrid Training (2025)
**Paper**: Nature Scientific Reports - "Integrating convolutional layers and biformer network with forward-forward and backpropagation training"

**Approach**:
- Three stages:
  1. Feature extraction with CNNs
  2. Attention refinement with BiFormer + FF algorithm
  3. Prediction network fine-tuning with backpropagation
- Applied to molecular property prediction

**Key Insight**: Uses FF for specific modules (attention) and backprop for others (prediction head) **simultaneously**

**Results**: Effective for domain-specific tasks

#### 3. Block-wise Hybrid Approaches
**Concept**:
- Divide network into blocks (e.g., ResNet residual blocks)
- Apply backprop within blocks
- Use FF or no gradients between blocks

**Results**: Can outperform pure backprop baselines while maintaining similar training speed

#### 4. Self-Contrastive Forward-Forward (SCFF, 2024-2025)
**Paper**: Nature Communications - "Self-Contrastive Forward-Forward algorithm"

**Approach**:
- Improved FF variant with self-contrastive learning
- Supports greedy layer-wise training and joint training
- Explicitly designed for unsupervised pretraining

**Results**:
- MNIST: 98.70% ± 0.01% (comparable to backprop)
- STL-10: 77.30% ± 0.12% (beats backprop: 77.02% ± 0.22%)
- **Explicitly noted**: "highly suitable for unsupervised pretraining followed by supervised BP training"

**Significance**: Directly validates our proposed approach!

### What's Missing: Sequential Phased Hybrid

**Existing hybrids** use FF and backprop:
- Simultaneously (Deep-CBN)
- In shallow integrated form (Integrated FF)
- Within/between blocks

**Our approach** is distinct:
- **Phase 1**: Pure FF for unsupervised representation learning (pretraining)
- **Phase 2**: Pure backprop for supervised fine-tuning (transfer learning)
- Clear temporal and architectural separation
- Focus on representation quality and transfer learning

This sequential phased approach has **not been systematically explored** in the literature.

---

## Performance Analysis

### Original Forward-Forward Performance

**MNIST** (Hinton, 2022):
- FF: ~1.4% test error
- Backprop: ~1.0% test error
- Gap: ~0.4% (modest)
- Note: FF requires 60 epochs vs backprop's 20 epochs

**CIFAR-10** (Hinton, 2022):
- FF: Competitive but slightly worse than backprop
- Performance gap increases with complexity
- More layers don't worsen gap significantly

### Recent Improvements

**ASGE (Adaptive Spatial Goodness Encoding, 2024)**:
- MNIST: 99.65%
- Fashion-MNIST: 93.41%
- CIFAR-10: 90.62%
- CIFAR-100: 65.42%
- Significant improvement over original FF

**Self-Contrastive FF (2024)**:
- MNIST: 98.70% ± 0.01% (near backprop parity)
- STL-10: 77.30% ± 0.12% (exceeds backprop)

**Contrastive Forward-Forward for Vision Transformers (2025)**:
- Up to 10% accuracy improvement over baseline FF
- 5-20x faster convergence
- Can outperform backprop under noisy supervision

### Key Performance Insights

1. **Representation Learning**: FF learns different representations than backprop
   - Focuses more on boundaries
   - Drops information unnecessary for classification decisions
   - This harms transfer learning in pure FF

2. **Training Efficiency**:
   - FF typically requires 2-3x more epochs
   - But each epoch is computationally lighter (no backward pass)
   - Memory footprint is lower

3. **Supervised vs Unsupervised**:
   - FF performs comparably in supervised settings
   - Transfer learning performance lags significantly
   - This is exactly where our hybrid approach may help!

---

## Common Experimental Setups

### Standard Architectures

**Fully Connected Networks**:
```
Input → FC(2000) → ReLU → FC(2000) → ReLU → FC(2000) → ReLU → FC(2000) → ReLU → FC(10)
```
- Common for MNIST/Fashion-MNIST
- 2-4 hidden layers
- 500-2000 neurons per layer

**Convolutional Networks**:
```
Conv Blocks → FF Training
FC Classifier → Supervised Training
```
- Challenge: FF doesn't naturally work with pooling
- Solutions: Spatial goodness encoding, channel-wise grouping

### Standard Datasets

1. **MNIST** (28×28 grayscale, 10 classes)
   - Most common benchmark
   - Easy label embedding (top-left pixels)
   - Quick iteration

2. **Fashion-MNIST** (28×28 grayscale, 10 classes)
   - More challenging than MNIST
   - Same structure for compatibility

3. **CIFAR-10** (32×32 RGB, 10 classes)
   - Standard benchmark for comparing to backprop
   - Tests scalability

4. **CIFAR-100** (32×32 RGB, 100 classes)
   - Tests fine-grained classification

5. **STL-10** (96×96 RGB, 10 classes)
   - Designed for unsupervised learning
   - Good for pretraining evaluation

### Training Protocols

**Standard FF Training**:
1. Generate positive samples with correct labels
2. Generate negative samples with incorrect labels
3. Forward pass positive samples → maximize goodness
4. Forward pass negative samples → minimize goodness
5. Update weights locally per layer

**Hyperparameters**:
- Learning rate: 0.001-0.03
- Threshold: 2.0-4.0 (for goodness function)
- Optimizer: Adam (most common)
- Batch size: 64-512
- Epochs: 60-250 for FF (vs 20-100 for backprop)

### Evaluation Metrics

**Primary**:
- Test accuracy
- Training time (wall-clock)
- Number of epochs to convergence

**Representation Quality**:
- Linear probing accuracy
- t-SNE/UMAP visualization of learned features
- Layer-wise analysis

**Efficiency**:
- Memory usage
- FLOPs per epoch
- Energy consumption (hardware-specific)

---

## Our Contribution

### Hypothesis
**FF-pretrained representations + backprop fine-tuning = improved performance and sample efficiency**

### Key Research Questions

1. **Does FF pretraining help supervised learning?**
   - Can FF learn useful unsupervised representations?
   - Do these representations transfer well to supervised tasks?

2. **What is the optimal architecture split?**
   - How many layers should use FF?
   - What's the role of network depth?

3. **Training protocol optimization**
   - Should FF layers be frozen during backprop phase?
   - How many FF iterations are needed?
   - What's the best fine-tuning strategy?

4. **Performance vs. pure approaches**
   - Hybrid vs. pure backprop (accuracy, speed, sample efficiency)
   - Hybrid vs. pure FF (accuracy, convergence)
   - Hybrid vs. random init + backprop (value of pretraining)

### Specific Contribution: Sequential Phased Hybrid

**Phase 1: FF Pretraining** (Unsupervised)
- Train first N layers with Forward-Forward
- Use data augmentation for positive samples
- Use corrupted/augmented data for negative samples
- No label information required
- Focus on representation learning

**Phase 2: Backprop Fine-tuning** (Supervised)
- Add classifier head on top
- Use supervised labels
- Options:
  a) Freeze FF layers, train classifier only
  b) Fine-tune all layers with backprop
  c) Fine-tune subset of layers

**Phase 3: Evaluation**
- Compare against baselines
- Analyze representation quality
- Study layer-wise feature evolution

### Why This Matters

1. **Novel experimental paradigm**: Clear separation between unsupervised and supervised phases

2. **Practical relevance**: Mimics common transfer learning workflows

3. **Theoretical insight**: Tests whether FF learns useful representations despite poor transfer performance in pure FF

4. **Bridges communities**: Connects biologically plausible learning (FF) with practical deep learning (backprop)

### Expected Outcomes

**Optimistic**:
- Hybrid outperforms pure backprop in low-data regime
- FF pretraining provides useful inductive bias
- Better generalization, especially with data augmentation

**Realistic**:
- Hybrid comparable to pure backprop but with different computational tradeoffs
- Insights into what makes FF representations transferable
- Identification of optimal split points

**Pessimistic**:
- Pure backprop still superior
- But: negative result still informative about limitations of local learning for transfer

### Connection to Existing Work

**SCFF explicitly suggests our approach**:
> "SCFF is highly suitable for unsupervised pretraining followed by supervised BP training"

Our work systematically explores this claim with:
- Multiple architectures and split points
- Comprehensive baseline comparisons
- Thorough ablation studies
- Analysis of learned representations

---

## References

### Core Papers

1. **Hinton, G. (2022)**. "The Forward-Forward Algorithm: Some Preliminary Investigations."
   arXiv:2212.13345
   https://arxiv.org/abs/2212.13345

2. **Gandhi, V., et al. (2023)**. "Extending the Forward Forward Algorithm."
   arXiv:2307.04205
   https://arxiv.org/abs/2307.04205

3. **Author Unknown (2023)**. "The Integrated Forward-Forward Algorithm: Integrating Forward-Forward and Shallow Backpropagation With Local Losses."
   arXiv:2305.12960
   https://arxiv.org/abs/2305.12960

4. **Chen, H., et al. (2024)**. "Self-Contrastive Forward-Forward algorithm."
   Nature Communications
   https://www.nature.com/articles/s41467-025-61037-0

5. **Scodellaro, D., et al. (2024)**. "Training Convolutional Neural Networks with the Forward-Forward Algorithm."
   arXiv:2312.14924
   https://arxiv.org/abs/2312.14924

6. **Various Authors (2025)**. "Integrating convolutional layers and biformer network with forward-forward and backpropagation training."
   Nature Scientific Reports
   https://www.nature.com/articles/s41598-025-92218-y

### Self-Supervised and Representation Learning

7. **Author Unknown (2023)**. "A Study of Forward-Forward Algorithm for Self-Supervised Learning."
   arXiv:2309.11955
   https://arxiv.org/abs/2309.11955

8. **Author Unknown (2024)**. "Adaptive Spatial Goodness Encoding: Advancing and Scaling Forward-Forward Learning Without Backpropagation."
   arXiv:2509.12394
   https://arxiv.org/abs/2509.12394

### Biological Plausibility and Local Learning

9. **Baldi, P., et al. (2016)**. "A theory of local learning, the learning channel, and the optimality of backpropagation."
   Neural Networks, Volume 83
   https://arxiv.org/abs/1506.06472

### Recent Extensions (2024-2025)

10. **Various Authors (2024)**. "On Advancements of the Forward-Forward Algorithm."
    arXiv:2504.21662

11. **Various Authors (2024)**. "Going Forward-Forward in Distributed Deep Learning."
    arXiv:2404.08573

12. **Various Authors (2025)**. "Contrastive Forward-Forward: A Training Algorithm of Vision Transformer."
    arXiv:2502.00571

13. **Various Authors (2025)**. "Scalable Forward-Forward Algorithm."
    arXiv:2501.03176

---

## Implementation Resources

### GitHub Repositories

- **Official PyTorch Examples**: pytorch/examples/mnist_forward_forward
- **mpezeshki/pytorch_forward_forward**: Clean PyTorch implementation
- **carloalbertobarbano/forward-forward-pytorch**: With hard negatives
- **loeweX/Forward-Forward**: Comprehensive reimplementation

### Tutorials

- **Keras Official Tutorial**: https://keras.io/examples/vision/forwardforward/
- **SNNTorch Tutorial**: FF with Spiking Neural Networks

---

## Next Steps

Based on this research, we are ready to:

1. Design a focused experimental protocol
2. Implement the sequential phased hybrid approach
3. Evaluate against comprehensive baselines
4. Analyze what makes FF representations transferable (or not)
5. Document findings and contribute to the understanding of hybrid training schemes

The literature review confirms:
- ✅ FF is well-understood
- ✅ Multiple implementations exist
- ✅ Some hybrid work exists but not our specific approach
- ✅ SCFF explicitly recommends FF pretraining → BP fine-tuning
- ✅ This is a tractable research question with clear experimental design
