# In-Training-Distribution Knowledge Classification via UNKNOWN Token Sink

## Project Overview

This research prototype explores a novel approach to improving language model calibration and reducing hallucinations through the introduction of a special `<UNKNOWN>` token that serves as an explicit abstention mechanism.

## Core Idea

The central hypothesis is that language models can be trained to abstain from making predictions on out-of-distribution or semantically meaningless inputs by learning to output a special `<UNKNOWN>` token, rather than hallucinating plausible-sounding but incorrect responses.

### Key Concept

Modify LLM training to include a special `<UNKNOWN>` token that serves as a sink for any concepts or inputs not seen in the training distribution. Train the model to prefer outputting `<UNKNOWN>` on out-of-distribution or gibberish inputs rather than hallucinating a specific prediction.

### Training Approach

Use synthetic gibberish data to train this behavior:
- **Repeating words**: "apple apple apple apple"
- **Random token sequences**: Incoherent combinations of valid tokens
- **Semantically null inputs**: Grammatically correct but meaningless text
- **Corrupted data**: Real text with random token replacements

The model learns to associate these patterns with the `<UNKNOWN>` token output, creating an explicit pathway for abstention when facing uncertainty or nonsense.

## Technical Stack

- **Language**: Python
- **Package Manager**: uv
- **Deep Learning Framework**: PyTorch
- **Model Library**: Hugging Face Transformers
- **Additional Tools**: Datasets, Tokenizers, Accelerate

## Research Question

Can an explicit abstention token, trained specifically on synthetic gibberish data, improve model calibration and reduce hallucinations on out-of-distribution inputs without degrading in-distribution performance?

## Hypothesis

By providing a dedicated "escape hatch" and training the model to recognize patterns of nonsense or unfamiliarity, we can:
1. Reduce hallucination rates on OOD data
2. Improve calibration (alignment between confidence and correctness)
3. Enable more reliable deployment by making uncertainty explicit
4. Maintain or improve in-distribution performance

## Success Criteria

1. High `<UNKNOWN>` usage on synthetic gibberish (>90%)
2. Low `<UNKNOWN>` usage on in-distribution data (<5%)
3. Appropriate `<UNKNOWN>` usage on real OOD data (TBD based on dataset)
4. Minimal degradation of in-distribution accuracy (<5% relative to baseline)
5. Measurable reduction in hallucination rate compared to baseline

## Novelty

While abstention and uncertainty quantification have been explored in ML, this approach uniquely combines:
- An explicit learnable abstention token
- Synthetic gibberish data for training
- Direct integration into the model's vocabulary and training objective
- Focus on LLMs specifically (not just classification tasks)

## Potential Challenges

1. **Token Integration**: How to properly add `<UNKNOWN>` to existing tokenizers
2. **Training Dynamics**: Balancing gibberish training with normal data
3. **Over-abstention**: Risk of model becoming too conservative
4. **Under-abstention**: Risk of not generalizing beyond synthetic gibberish
5. **Evaluation**: Defining appropriate metrics and OOD datasets
6. **Multi-token Outputs**: Handling sequences, not just single classifications

## Project Status

This is an early-stage research prototype. Current phase: Literature review and feasibility assessment.
