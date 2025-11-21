# Deep Research: Intermixed Forward-Forward and Backpropagation Layers

## Executive Summary

This document provides comprehensive research on **architecturally intermixed** Forward-Forward (FF) and backpropagation (BP) layers—where different layers within the same network use different training algorithms **simultaneously** during the forward pass.

**Key Finding**: Several approaches exist that intermix local and global learning within the same network architecture, but most maintain some form of separation. True layer-by-layer intermixing (e.g., Layer 1: FF, Layer 2: BP, Layer 3: FF, Layer 4: BP) remains largely unexplored.

---

## Table of Contents
1. [Existing Intermixed Architectures](#existing-intermixed-architectures)
2. [Block-wise Hybrid Approaches](#block-wise-hybrid-approaches)
3. [Auxiliary Classifiers and Local Losses](#auxiliary-classifiers-and-local-losses)
4. [Related Approaches](#related-approaches)
5. [Technical Challenges](#technical-challenges)
6. [Novelty Assessment](#novelty-assessment)
7. [Implementation Considerations](#implementation-considerations)

---

## Existing Intermixed Architectures

### 1. Deep-CBN (2025) - Most Relevant Example

**Paper**: "Integrating convolutional layers and biformer network with forward-forward and backpropagation training" (Nature Scientific Reports)
**GitHub**: akianfar/Deep-CBN

#### Architecture Details

**Three-Stage Pipeline**:
```
Input (SMILES strings)
    ↓
[Stage 1: CNN Feature Extraction]
    - Convolutional layers
    - Extract local molecular features
    - Training method: Not explicitly specified (likely standard)
    ↓
[Stage 2: BiFormer Attention with FF]
    - BiFormer attention mechanism
    - Captures global contextual information
    - Training method: **Forward-Forward algorithm**
    ↓
[Stage 3: Prediction Subnetwork]
    - Final prediction layers
    - Training method: **Backpropagation fine-tuning**
    ↓
Output (Molecular property predictions)
```

#### Key Insights
- **Sequential modular design**: Each stage has a distinct role
- **FF for attention mechanisms**: Uses FF where global context matters
- **BP for final prediction**: Uses BP for supervised fine-tuning
- **Performance**: ROC-AUC of 0.992 on ClinTox (vs 0.945 for previous methods)

#### How It Works
1. CNN layers encode molecular features from raw SMILES strings
2. BiFormer attention module (trained with FF) captures long-range dependencies
3. Prediction head (trained with BP) maps representations to properties
4. **Hybrid training**: FF and BP are applied to different modules

#### Limitations
- Domain-specific (molecular property prediction)
- Modular separation (not fine-grained layer intermixing)
- No analysis of FF/BP interface challenges

---

### 2. Scalable Forward-Forward (SFF, 2025) - Block-wise Hybrid

**Paper**: arXiv:2501.03176 - "Scalable Forward-Forward Algorithm"

#### Architecture Details

**Block-wise Design**:
```
Network = [Block_1, Block_2, ..., Block_N, Classifier]

Each Block:
    - Multiple conv/FC layers
    - **Internal backpropagation allowed**
    - Auxiliary goodness predictor attached

Between Blocks:
    - **No gradient flow** (FF-style local optimization)
    - Each block optimized independently

Classifier:
    - Aggregates goodness scores from all blocks
    - Final prediction
```

#### Key Innovation
- **Hybrid approach**: "Backpropagation within each model block, but not between blocks"
- **Preserves modularity**: Each block can be trained independently
- **Memory efficient**: No need to store activations across blocks

#### Performance Results
| Dataset | Architecture | SFF | Backprop |
|---------|-------------|-----|----------|
| CIFAR-10 | CNNB | 81.38% | 78.49% |
| Imagenette | ResNet18 | 83.24% | 81.84% |

**Key Finding**: "This hybrid design tends to outperform backpropagation baselines while maintaining similar training speed."

#### Technical Details
1. **Auxiliary layers** compute class-specific goodness per block
2. **Local loss computation** happens independently per block
3. **No cross-block gradients** - parameters updated locally
4. **Quote**: "Whose output is used only for the computation of the local loss and will not be passed to the next layer"

---

### 3. Layer Collaboration in Forward-Forward (2024)

**Paper**: AAAI-24 - "Layer Collaboration in the Forward-Forward Algorithm" (arXiv:2305.12393)
**Authors**: Lorberbom, Gat, Adi, Schwing, Hazan

#### Problem Addressed
- Original FF trains each layer independently
- Lack of collaboration limits learning complex features
- Information flow between layers is suboptimal

#### Solution
- **Still pure FF** (no backpropagation mixed in)
- Enables layer collaboration through improved information flow
- Maintains local learning while allowing layers to coordinate

#### Key Quote
> "The current version of the forward-forward algorithm is suboptimal when considering information flow in the network, resulting in a lack of collaboration between layers."

#### Relevance
- Shows FF layers can be made to collaborate
- Does NOT intermix with backpropagation
- Theoretical grounding in functional entropy theory
- No additional computational overhead

---

## Block-wise Hybrid Approaches

### Conceptual Framework

**Block-wise Training Pattern**:
```
Network Structure:
    Block 1: [Layer 1, Layer 2, Layer 3]
             ↓ (BP within)
             Local objective + auxiliary classifier
             ↓ (NO BP across)
    Block 2: [Layer 4, Layer 5, Layer 6]
             ↓ (BP within)
             Local objective + auxiliary classifier
             ↓ (NO BP across)
    Block 3: [Layer 7, Layer 8, Layer 9]
             ↓ (BP within)
             Global classifier
```

### Advantages
1. **Reduced memory**: No need for full backward pass storage
2. **Modularity**: Blocks can be trained independently or in parallel
3. **Flexibility**: Can integrate non-differentiable components
4. **Biological plausibility**: More aligned with local learning

### Disadvantages
1. **Sub-optimal gradients**: Local objectives may not align with global objective
2. **Performance gap**: Often slightly worse than end-to-end BP (though SFF challenges this)
3. **Hyperparameter complexity**: Need to tune block sizes and local objectives

---

## Auxiliary Classifiers and Local Losses

### GoogLeNet / Inception (2014) - Historical Precedent

**Architecture Pattern**:
```
Input
  ↓
[Early Conv Blocks]
  ↓
[Auxiliary Classifier 1] ← Local supervised loss (0.3 weight)
  ↓
[Middle Conv Blocks]
  ↓
[Auxiliary Classifier 2] ← Local supervised loss (0.3 weight)
  ↓
[Later Conv Blocks]
  ↓
[Main Classifier] ← Global supervised loss (1.0 weight)
```

### Purpose
- Combat vanishing gradients in deep networks (22 layers)
- Provide direct supervision to intermediate layers
- Encourage discriminative features early in the network

### Training Method
- **All backpropagation**: Auxiliary losses are summed with weights
- Total loss = 1.0 × main_loss + 0.3 × aux_loss_1 + 0.3 × aux_loss_2
- Gradients flow through entire network (not local learning)

### Key Difference from FF/BP Hybrid
- **Not a hybrid**: All layers use backpropagation
- **Multiple losses**: But all losses are differentiable and backpropped
- **Inspiration**: Shows multi-objective training can work

---

## Related Approaches

### 1. Greedy Layer-wise Pretraining (2006-2007)

**Historical Context**: Pioneered deep learning before modern techniques

**Two-Phase Process**:
```
Phase 1: Unsupervised Pretraining (Layer-wise)
    Layer 1: Pretrain with local unsupervised objective (e.g., RBM, autoencoder)
    Layer 2: Pretrain using Layer 1's output
    Layer 3: Pretrain using Layer 2's output
    ...

Phase 2: Supervised Fine-tuning (End-to-end)
    All layers: Fine-tune with backpropagation on supervised task
```

**Key Characteristics**:
- **Sequential phases**: Not intermixed during training
- **All-BP fine-tuning**: Phase 2 uses standard backpropagation
- **Historical note**: Largely abandoned except in NLP (transformers use pretraining)

**Why It's Different from Our Approach**:
- Temporal separation (pretrain then fine-tune)
- Not architectural intermixing (all layers use same method in each phase)
- Modern versions use self-supervised objectives, not local learning rules

### 2. Target Propagation (2014-2020)

**Core Idea**: Replace backward pass with forward pass using target generation

**Method**:
```
Forward pass: x → h₁ → h₂ → h₃ → y
                                ↓
                           Compute loss
                                ↓
Target generation (backward): y_target ← h₃_target ← h₂_target ← h₁_target
                                         ↓
Local losses: Each layer minimizes ||h_i - h_i_target||
```

**Relation to FF/BP Intermixing**:
- Uses local losses like FF
- But all layers use same method (target propagation)
- Not a hybrid with backpropagation

**Challenges**:
- "Difference target propagation struggles to select good local targets in the upstream layers"
- Performance generally below backpropagation

### 3. Decoupled Neural Interfaces / Synthetic Gradients (2016-2017)

**Paper**: DeepMind - "Decoupled Neural Interfaces using Synthetic Gradients" (arXiv:1608.05343)

**Core Problem**: Update locking in backpropagation
- Layers must wait for forward pass to complete
- Then wait for backward pass to reach them
- Prevents asynchronous or parallel layer updates

**Solution**: Predict gradients locally
```
Forward pass:
    x → Layer 1 → h₁ → Layer 2 → h₂ → Layer 3 → y
                   ↓                ↓              ↓
    Synthetic Gradient Modules predict gradients locally
                   ↓                ↓              ↓
    Update Layer 1    Update Layer 2    Update Layer 3
    (without waiting for true backprop gradients)
```

**Key Insight**:
- Each layer trains a model to predict what gradient it will receive
- Allows asynchronous updates (no locking)
- "Surprisingly lead to drastically different layer-wise representations"

**Relevance**:
- Different learning dynamics per layer
- But still fundamentally backprop-based (predicting gradients)
- Shows heterogeneous layer training is possible

**Challenge**:
- "Quality of loss estimation degrades towards the bottom when multiple SGs bootstrap from each other"

### 4. Neural Architecture Search with Mixed Bio-inspired Learning Rules (2024)

**Paper**: arXiv:2507.13485

**Core Finding**:
> "Neural networks using different bio-inspired learning rules for different layers achieve better accuracy than those using a single rule across all layers."

**Performance**:
- Mixed rules: 95.16% on CIFAR-10, 76.48% on CIFAR-100
- Shows heterogeneous layer training can improve performance

**Method**:
- NAS searches over both architectures AND learning rules per layer
- Different layers get different bio-inspired rules (Hebbian, STDP, etc.)

**Relevance**:
- Direct evidence that layer-specific learning rules can help
- But uses bio-inspired rules, not FF specifically
- Validates the concept of intermixed training methods

---

## Technical Challenges

### 1. Gradient Flow at FF/BP Interfaces

**The Core Problem**: How do gradients flow through FF layers during backpropagation?

**Challenge Details**:
- FF layers don't compute gradients in the traditional sense
- FF layers use local goodness functions, not differentiable outputs
- During BP phase, gradients need to flow backward through FF layers

**Possible Solutions**:

**Option A: Freeze FF Layers**
```python
# Freeze FF layers during BP training
for param in ff_layers.parameters():
    param.requires_grad = False

# Only train BP layers
bp_layers.train()
```
- Pros: Simple, no gradient issues
- Cons: FF layers can't adapt to supervised task

**Option B: Fine-tune FF Layers with BP**
```python
# Treat FF layers as normal layers during BP
# Ignore FF training, use them as initialized weights
entire_network.train()  # All layers
optimizer = Adam(entire_network.parameters())
```
- Pros: Full end-to-end optimization
- Cons: Loses FF characteristics, becomes standard BP

**Option C: Maintain Dual Training**
```python
# Continue FF training on FF layers
# Simultaneously run BP on BP layers
for epoch in range(num_epochs):
    # FF update for FF layers
    ff_positive_loss.backward()
    ff_negative_loss.backward()

    # BP update for BP layers
    supervised_loss.backward()  # Only affects BP layers
```
- Pros: Maintains both training paradigms
- Cons: Complex, may have conflicting updates

**Option D: Detach at Interface**
```python
# Forward pass
x = ff_layers(x)
x = x.detach()  # Stop gradients here
y = bp_layers(x)

# Backward pass only affects BP layers
loss = criterion(y, target)
loss.backward()  # Stops at detach point
```
- Pros: Clean separation, no gradient conflicts
- Cons: BP layers can't influence FF layers

### 2. Normalization and Information Bottleneck

**FF Requirement**: Layer normalization to prevent trivial solutions
```python
# FF needs normalized activations
h = ff_layer(x)
h_normalized = h / torch.norm(h)  # Prevent length-based discrimination
goodness = (h_normalized ** 2).sum()
```

**Challenge**: Normalization removes information
- "This removes all of the information that was used to determine the goodness in the first hidden layer"
- BP layers receive impoverished inputs from FF layers

### 3. Positive/Negative Sample Generation for FF Layers

**For supervised tasks**: Need both positive and negative samples

**Strategies**:
1. **Label-based** (requires labels):
   ```python
   positive = embed_label(x, correct_label)
   negative = embed_label(x, wrong_label)
   ```

2. **Augmentation-based** (unsupervised):
   ```python
   positive = augment(x, strong=False)
   negative = augment(x, strong=True) or corrupt(x)
   ```

**Challenge**: What if FF layers are in the middle of network?
- Can't easily embed labels in intermediate representations
- Augmentation may not make sense for high-level features

### 4. Loss Balancing and Convergence

**Multi-objective optimization**:
```python
total_loss = α * ff_loss_layer1 + β * ff_loss_layer2 + γ * bp_loss_final
```

**Challenges**:
- How to balance loss weights (α, β, γ)?
- FF and BP losses have different scales and meanings
- Risk of one objective dominating
- Convergence behavior unclear

---

## Novelty Assessment

### What HAS Been Done

#### ✅ Block-wise FF + Internal BP (SFF, 2025)
- BP within blocks, FF between blocks
- Proven effective on CIFAR-10, Imagenette
- Clear methodology and results

#### ✅ Modular Stage-wise Hybrid (Deep-CBN, 2025)
- CNN → FF attention → BP predictor
- Domain-specific (molecular prediction)
- Sequential module design

#### ✅ Auxiliary Classifiers with BP (GoogLeNet, 2014)
- Multiple local losses, all BP-based
- Proven to help with vanishing gradients
- Not true local learning

#### ✅ Greedy Layer-wise Pretrain + BP Fine-tune (2006-2007)
- Sequential phases, not simultaneous
- Largely superseded by modern techniques

### What HAS NOT Been Systematically Explored

#### ❌ Fine-grained Layer-by-Layer Intermixing
Pattern like:
```
Input → [FF Layer 1] → [FF Layer 2] → [BP Layer 3] → [BP Layer 4] → Output
```
Or:
```
Input → [FF] → [BP] → [FF] → [BP] → [FF] → [BP] → Output
```

**Why it's different**:
- Not block-wise (single layers, not blocks)
- Not modular stages (intermixed throughout)
- True fine-grained heterogeneity

**Why it hasn't been done**:
- Technical complexity (gradient flow challenges)
- Unclear benefits over simpler approaches
- Difficult to justify the added complexity

#### ❌ Systematic Study of FF Depth Variation
Questions like:
- How many FF layers is optimal?
- Does FF→BP→FF→BP help?
- What's the effect of split point depth?

**Some exploration** in SFF (block sizes), but not for individual layers.

#### ❌ Adaptive/Learned Intermixing
- Let the network learn which layers should use FF vs BP
- Neural architecture search over training methods per layer
- Mentioned in "Mixed Bio-inspired Learning Rules" but not with FF

---

## Implementation Considerations

### Approach 1: Sequential Phased (Safest)

**Architecture**:
```
Phase 1: FF pretraining
    Layers 1-N: Train with FF

Phase 2: BP fine-tuning
    Layers 1-N: Freeze or fine-tune
    Layer N+1 (classifier): Train with BP
```

**Pros**:
- Clean separation, no gradient conflicts
- Well-precedented (like greedy pretraining)
- Easy to implement and debug

**Cons**:
- Not truly "intermixed" during training
- Can't leverage both methods simultaneously

### Approach 2: Block-wise Hybrid (Proven)

**Architecture**:
```
Block 1: [Layer 1-3] with internal BP, external FF
Block 2: [Layer 4-6] with internal BP, external FF
Block 3: [Layer 7-9] with internal BP, external FF
Classifier: Aggregate goodness scores
```

**Pros**:
- SFF proved this works and can beat pure BP
- Modular and memory efficient
- Balance between FF benefits and BP effectiveness

**Cons**:
- Not layer-level intermixing
- Requires auxiliary classifiers per block
- More complex training loop

### Approach 3: Detached Interface (Novel)

**Architecture**:
```
FF Layers (1-N):
    - Train with FF (positive/negative passes)
    - Detach gradients at output

BP Layers (N+1 to end):
    - Take FF output (detached)
    - Train with standard BP
    - No gradients flow to FF layers
```

**Pros**:
- True simultaneous training
- No gradient conflicts (clean detach)
- FF layers maintain their training characteristics

**Cons**:
- FF layers can't adapt to BP layers' needs
- May lead to suboptimal interfaces
- Unclear if FF layers will learn useful features

### Approach 4: Dual-objective (Experimental)

**Architecture**:
```
All layers simultaneously:
    - FF layers: Optimize local goodness
    - BP layers: Optimize global loss
    - FF layers: ALSO receive BP gradients

Combined optimization:
    θ_ff ← θ_ff - α∇L_ff - β∇L_bp
```

**Pros**:
- True hybrid optimization
- Layers can balance local and global objectives

**Cons**:
- Very complex to implement correctly
- Conflicting gradients may interfere
- No precedent in literature
- Likely to be unstable

---

## Recommended Experimental Progression

### Stage 1: Sequential Phased Baseline
**Goal**: Establish baseline hybrid performance

```
Architecture: [FF Layer 1] → [FF Layer 2] → [FF Layer 3] → [BP Classifier]
Training:
    1. Phase 1: Train FF layers (100 epochs)
    2. Phase 2a: Freeze FF, train classifier (50 epochs)
    3. Phase 2b: Fine-tune all with BP (50 epochs)
```

**Baselines**: Pure BP, Pure FF, Random init + BP classifier

### Stage 2: Block-wise Hybrid
**Goal**: Test SFF-style block-wise approach

```
Architecture: [FF Block 1] → [FF Block 2] → [BP Classifier]
Block 1: 2 layers with internal BP
Block 2: 2 layers with internal BP
Between blocks: No gradient flow
```

**Comparison**: Sequential phased vs block-wise

### Stage 3: Detached Interface (Novel)
**Goal**: True simultaneous FF+BP training

```
Architecture: [FF Layers 1-3] → DETACH → [BP Layers 4-5]
Training: Simultaneous FF and BP updates
```

**Analysis**:
- Compare frozen vs detached FF layers
- Measure representation quality at interface
- Test different split points

### Stage 4: Variations (If promising)
- Multiple FF→BP→FF→BP alternations
- Different FF/BP ratios
- Learned layer-wise training method selection

---

## Conclusions

### Key Findings

1. **Block-wise hybrids exist and work** (SFF, 2025)
   - Proven to match or exceed pure BP
   - Good balance of efficiency and effectiveness

2. **Modular stage-wise hybrids exist** (Deep-CBN, 2025)
   - Effective for domain-specific tasks
   - Clear architectural separation

3. **Fine-grained layer-by-layer intermixing is unexplored**
   - No systematic study in literature
   - Technical challenges are significant
   - Unclear if benefits justify complexity

4. **Related work provides valuable insights**
   - Auxiliary classifiers: Local losses can help
   - Synthetic gradients: Layers can be decoupled
   - Mixed learning rules: Heterogeneity can improve performance

### Novelty of Proposed Approach

**High Novelty**:
- Layer-by-layer FF/BP intermixing (e.g., FF→BP→FF→BP)
- Adaptive/learned training method per layer
- Systematic study of split point effects

**Medium Novelty**:
- Sequential phased FF pretrain → BP fine-tune (SCFF suggests it, but no systematic study)
- Detached interface simultaneous training
- Multiple split points in deep networks

**Low Novelty**:
- Block-wise hybrid (SFF already did this)
- Modular stage-wise (Deep-CBN already did this)

### Recommendations

**For Research Value**:
1. Start with **sequential phased** (safest, well-motivated)
2. Compare with **block-wise hybrid** (proven effective)
3. Experiment with **detached interface** (novel, interesting)
4. Analyze **representation quality** at interfaces (key insight)

**For Practical Value**:
1. Focus on **block-wise hybrid** (SFF shows it works)
2. Optimize for **memory efficiency** (FF's main advantage)
3. Test on **larger scales** than SFF (they stopped at ResNet18)

**For Theoretical Insight**:
1. Study **gradient flow** at interfaces
2. Analyze **layer-wise representations** (t-SNE, CKA)
3. Understand **when and why** FF helps BP
4. Investigate **optimal split points** theoretically

---

## References

### Core Papers on Intermixed Architectures

1. **Scalable Forward-Forward Algorithm** (2025)
   arXiv:2501.03176
   - Block-wise BP within, FF between
   - Best evidence that hybrids can outperform pure BP

2. **Deep-CBN** (2025)
   Nature Scientific Reports - "Integrating convolutional layers and biformer network with forward-forward and backpropagation training"
   GitHub: akianfar/Deep-CBN
   - Modular stage-wise hybrid
   - FF for attention, BP for prediction

3. **Layer Collaboration in Forward-Forward** (2024)
   AAAI-24, arXiv:2305.12393
   - Improves FF layer cooperation
   - Pure FF, no BP intermixing

### Related Approaches

4. **Decoupled Neural Interfaces** (2016)
   arXiv:1608.05343 (DeepMind)
   - Synthetic gradients for async training
   - Shows layers can be decoupled

5. **Mixed Bio-inspired Learning Rules** (2024)
   arXiv:2507.13485
   - NAS over learning rules per layer
   - Evidence for heterogeneous training

6. **Greedy Layer-wise Training** (2006)
   Hinton et al., Bengio et al.
   - Historical precedent for phased training
   - Unsupervised pretrain + BP fine-tune

### Auxiliary Methods

7. **GoogLeNet / Inception** (2014)
   - Auxiliary classifiers at intermediate layers
   - All BP, but multi-objective

8. **Target Propagation** (2014-2020)
   - Local target-based learning
   - Alternative to BP, struggles with early layers

---

## Appendix: Gradient Flow Mathematics

### Standard Backpropagation
```
Forward: x → h₁ → h₂ → h₃ → y
         W₁    W₂    W₃    W₄

Backward:
∂L/∂W₄ = ∂L/∂y · ∂y/∂W₄
∂L/∂W₃ = ∂L/∂y · ∂y/∂h₃ · ∂h₃/∂W₃
∂L/∂W₂ = ∂L/∂y · ∂y/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂W₂
∂L/∂W₁ = ∂L/∂y · ∂y/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁
```

### Forward-Forward (Local)
```
Layer i:
    Forward positive: x+ → hᵢ⁺
    Forward negative: x- → hᵢ⁻

    Goodness: g(hᵢ) = Σ(hᵢ²)

    Loss: Lᵢ = log(1 + exp(-g(hᵢ⁺) + θ)) + log(1 + exp(g(hᵢ⁻) - θ))

    Update: ∂Lᵢ/∂Wᵢ only (no chain rule beyond this layer)
```

### Hybrid Option A: Detached Interface
```
FF Layers: x → h₁[FF] → h₂[FF] → h₂.detach() → h₃[BP] → y
                                      ↑
                                  Stop gradients

FF Update: ∂L_FF/∂W₁, ∂L_FF/∂W₂ (local)
BP Update: ∂L_BP/∂W₃, ∂L_BP/∂W₄ (from y, stops at detach)
```

### Hybrid Option B: Dual Objective
```
Total Loss: L_total = α·L_FF + β·L_BP

∂L_total/∂W₁ = α·∂L_FF/∂W₁ + β·∂L_BP/∂W₁  (conflicting?)
∂L_total/∂W₂ = α·∂L_FF/∂W₂ + β·∂L_BP/∂W₂  (conflicting?)
∂L_total/∂W₃ = β·∂L_BP/∂W₃  (BP only)
```

Challenge: FF loss is local, BP loss chains through many layers. Do they cooperate or conflict?

---

**End of Intermixed Architectures Research Document**

*Last Updated: November 2025*
*Research conducted for: Forward Forward with Backprop project*
