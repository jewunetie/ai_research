# Forward Forward with Backprop

## Project Overview

This is a research prototype exploring **hybrid training schemes** that combine Hinton's Forward-Forward algorithm with standard backpropagation.

## Concept

The Forward-Forward (FF) algorithm, introduced by Geoffrey Hinton in 2022, is an alternative to backpropagation that uses local learning rules. This project investigates a specific hybrid approach:

1. **Phase 1: Unsupervised Representation Learning**
   - Use Forward-Forward algorithm to train early layers of a neural network
   - Learn representations through contrastive goodness functions
   - No gradient backpropagation through the network

2. **Phase 2: Supervised Fine-tuning**
   - Apply standard backpropagation to train later layers or a classifier head
   - Optionally fine-tune the Forward-Forward pretrained layers
   - Leverage learned representations for downstream supervised tasks

The goal is to explore whether this hybrid approach can:
- Combine the benefits of local learning (FF) with global optimization (backprop)
- Improve sample efficiency or generalization
- Provide insights into hierarchical representation learning
- Offer practical advantages in specific scenarios

## Technical Stack

- **Primary Language**: Python
- **Package Management**: uv
- **ML Framework**: PyTorch
- **Model/Data Access**: Hugging Face (where applicable)

## Project Status

This is a **research prototype** designed to test a specific hypothesis about hybrid training approaches. The implementation prioritizes:
- Clear experimental design
- Reproducible results
- Systematic comparison with baselines
- Documentation of findings

## Research Questions

1. Can Forward-Forward pretraining improve downstream supervised performance?
2. What is the optimal split point between FF and backprop layers?
3. How does the hybrid approach compare to pure backprop in terms of:
   - Final accuracy
   - Training efficiency
   - Representation quality
   - Memory and compute requirements

## Next Steps

1. Literature review and feasibility check
2. Research summary and experimental design
3. Implementation of hybrid training pipeline
4. Empirical evaluation and analysis
