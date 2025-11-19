# Research Summary: UNKNOWN Token Sink for LLM Calibration

## Executive Summary

This document synthesizes research on abstention, calibration, and uncertainty quantification in large language models (LLMs) to contextualize the proposed UNKNOWN token sink approach. The research reveals a rich landscape of existing work on selective prediction and abstention, but identifies a gap: **no prior work systematically trains a dedicated abstention token using synthetic gibberish data**.

---

## 1. Abstention and Calibration Methods

### 1.1 Key Survey Papers

**"Know Your Limits: A Survey of Abstention in Large Language Models"**
- **Citation**: Wen et al., 2024, arXiv:2407.18418 (Published in TACL, MIT Press)
- **Key Contributions**:
  - Framework examining abstention from three perspectives: query, model, and human values
  - Comprehensive organization of abstention methods, benchmarks, and evaluation metrics
  - Recognizes abstention's potential to mitigate hallucinations and enhance safety
- **URL**: https://arxiv.org/abs/2407.18418

**"A Survey on Out-of-Distribution Detection in NLP"**
- **Citation**: Lang et al., 2023, TMLR
- **Key Contributions**:
  - Comprehensive overview of OOD detection methods for NLP
  - Critical for building safe NLP systems (classification, QA, translation)
  - Addresses violation of closed-world assumption in deployed models
- **URL**: https://arxiv.org/abs/2305.03236

**"A Survey on Uncertainty Quantification of Large Language Models"**
- **Citation**: 2024, arXiv:2412.05563
- **Key Contributions**:
  - Extensive overview with mechanistic interpretability discussion
  - Covers unique LLM uncertainty sources (input ambiguity, decoding stochasticity)
  - Broad applications beyond traditional UQ methods
- **URL**: https://arxiv.org/abs/2412.05563

### 1.2 Selective Prediction Approaches

**"The Art of Abstention: Selective Prediction and Error Regularization for NLP"**
- **Citation**: Xin & Tang, ACL 2021
- **Key Contributions**:
  - Foundational work on selective prediction for NLP
  - Proposes error regularization to improve confidence estimation
  - Framework: predictor function f(x) + selector function g(x)
- **URL**: https://aclanthology.org/2021.acl-long.84/
- **Limitation**: Often struggles in out-of-domain settings

**Selective "Selective Prediction": Reducing Unnecessary Abstention**
- **Citation**: ACL Findings 2024
- **Key Contributions**:
  - Introduces ReCoVERR algorithm for inference-time abstention reduction
  - Enables VLMs to answer up to 20% more questions without accuracy loss
  - Addresses over-abstention problem
- **URL**: https://aclanthology.org/2024.findings-acl.767/

**ASPIRE: Adaptation with Self-Evaluation to Improve Selective Prediction**
- **Citation**: Google Research, EMNLP Findings 2023
- **Key Contributions**:
  - Fine-tunes LLMs on QA via parameter-efficient tuning
  - Trains models to self-evaluate answer correctness
  - Outputs answer + confidence score; emits "I don't know" when confidence low
- **Performance**: Improved AUACC from 91.23% to 92.63% on CoQA
- **URL**: https://research.google/blog/introducing-aspire-for-selective-prediction-in-llms/
- **Relevance**: Most similar to our approach but uses confidence-based abstention rather than learned token

### 1.3 Calibration and Uncertainty Quantification

**"Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data"**
- **Citation**: EMNLP 2020, arXiv:2010.11506
- **Key Finding**: Fine-tuned pre-trained LMs suffer from severe miscalibration on both in-dist and OOD data due to over-parameterization
- **Solution**: Regularized fine-tuning method
- **Important Insight**: Larger pre-trained models are better calibrated and yield higher OOD detection performance
- **URL**: https://aclanthology.org/2020.emnlp-main.102/

**"LM-Polygraph: Benchmarking Uncertainty Quantification Methods for LLMs"**
- **Citation**: TACL 2024
- **Key Contributions**:
  - Novel multilingual automatic evaluation pipeline
  - Methods for normalized confidence scores with better calibration
  - Critical for selective classification where model abstains if confidence insufficient
- **URL**: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/

**"Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach"**
- **Citation**: 2024, arXiv:2404.15993
- **Key Finding**: Uncertainty estimation in small LMs may exhibit misalignment with prediction correctness
- **Important**: Well-calibrated uncertainty ≠ strong correlation with accuracy

**SPUQ: Sampling with Perturbation for Uncertainty Quantification**
- **Performance**: Reduced Expected Calibration Error by 50% on average
- **Demonstrates**: Potential to make LLMs more reliable through better UQ

**Conformal Prediction for Natural Language Processing**
- **Citation**: Survey, TACL 2024, arXiv:2405.01976
- **Key Contributions**:
  - Theoretically sound framework for uncertainty quantification
  - Model-agnostic and distribution-free (assumes only exchangeability)
  - Outputs prediction sets containing ground truth with user-specified probability
  - Used for conformal risk control to bound hallucination risk
- **Challenges**: Large output sets, recursive generation violates exchangeability
- **URL**: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/

### 1.4 Refusal and Rejection Mechanisms

**"Refusal Tokens: A Simple Way to Calibrate Refusals in LLMs"**
- **Citation**: arXiv:2412.06748
- **Key Approach**:
  - Prepends refusal tokens (one per category or single token) to responses during training
  - Enables steering refusal behavior at inference by modulating token probability
- **Relevance**: Most directly related to our proposal
- **Difference**: Uses refusal for safety/values alignment, not epistemic uncertainty
- **URL**: https://arxiv.org/abs/2412.06748

**"Do LLMs Know When to NOT Answer?"**
- **Citation**: arXiv:2407.16221
- **Focus**: Investigating abstention abilities of LLMs
- **Key Concept**: Abstention Ability (AA) - capability to withhold responses when uncertain

**Prompting-Based Abstention**
- **Approaches**:
  - Few-shot exemplars showing abstained vs. answered responses
  - Instruction hints: "Answer only if answerable", "Answer if safe"
  - Multiple-choice: Adding "None of the above" option
- **Limitation**: Requires careful prompt engineering, less reliable

---

## 2. Special Token Usage in Transformers

### 2.1 Standard Special Tokens

**[CLS] (Classification Token)**
- **Purpose**: Aggregates sequence-level information for classification
- **Mechanism**: Placed at beginning of sequence; output contains information from all tokens
- **Training**: BERT trained with [CLS] for next sentence prediction task

**[SEP] (Separator Token)**
- **Purpose**: Marks boundaries between sentences
- **Use Case**: Sequence pairs (classification, QA)

**[MASK] Token**
- **Purpose**: Masked language modeling during pre-training
- **Mechanism**: Random words masked; model learns to predict them
- **Training**: Core to BERT's MLM objective

### 2.2 Key Insights for Custom Tokens

**Training Requirements**:
- Special tokens have meaning because they were trained with the model
- Adding tokens changes `last_hidden_state`, affecting downstream tasks
- New tokens require sufficient training data and iterations to learn embeddings

**Integration Methods**:
- `tokenizer.add_tokens(new_tokens)` or `add_special_tokens()`
- `model.resize_token_embeddings(len(tokenizer))` to add embedding space
- New embeddings initialized randomly; require fine-tuning

### 2.3 Prior Work on Token Addition

**Domain-Specific Vocabulary Extension**
- **Use Cases**: Legal, medical, scientific documents
- **Challenge**: Balancing new tokens with existing subword tokenization
- **Best Practice**: Fine-tune on domain corpus after adding tokens

**Undertrained/Glitch Tokens**
- **Finding**: Some tokens in GPT models were in tokenizer but underrepresented in training data
- **Consequences**: Anomalous behavior, potential hallucinations
- **Lesson**: Token presence in vocabulary ≠ model understanding; training coverage critical

**Byte-Level Tokenization**
- **Example**: GPT-2 uses byte-level BPE
- **Advantage**: Base vocabulary of 256 bytes avoids unknown token issues entirely
- **Relevance**: Ensures every input can be tokenized, but doesn't address semantic understanding

---

## 3. Out-of-Distribution Detection Approaches

### 3.1 OOD Detection Methods

**Rejection Option for DNNs**
- **Concept**: Extend output space to include ® (reject/abstain) value
- **Application**: Detect and reject OOD inputs; use regular model for in-distribution
- **NLP Application**: Critical for open-world deployment vs. closed-world training assumption

**"Out-of-Distribution Detection and Selective Generation for Conditional Language Models"**
- **Citation**: arXiv:2209.15558
- **Key Contributions**:
  - Highly accurate, lightweight OOD detection for CLMs
  - Demonstrated on abstractive summarization and translation
  - Enables selective generation of high-quality outputs only
- **URL**: https://arxiv.org/abs/2209.15558

**Multi-Modal OOD Detection with LLMs**
- **Citation**: NeurIPS 2024, arXiv:2310.08027
- **Key Contribution**: Consistency-based uncertainty calibration for confidence estimation
- **Finding**: Active research at major venues (NeurIPS 2024)

### 3.2 Confidence Estimation Techniques

**Token Likelihood Methods**
- **Approach**: Use token probabilities to assess response uncertainty
- **Widespread**: Most common method for LLM uncertainty estimation
- **Limitation**: May not correlate well with correctness in small models

**Perplexity-Based Detection**
- **Metric**: Lower perplexity → greater confidence → lower hallucination chance
- **Application**: Can identify uncertain or OOD inputs

**Ensemble and Sampling Methods**
- **Approach**: Generate multiple outputs, assess consistency
- **Used in**: ASPIRE (answer sampling + self-evaluation)

---

## 4. Prior Work on "I Don't Know" Training

### 4.1 Explicit Abstention Training

**Knowledge-Based Refusals**
- **Approach**: Interrogate model to identify knowledge boundaries
- **Training**: Add "I don't know" examples to training data
- **Outcome**: Reduces hallucinations by allowing uncertainty expression
- **Phase**: Typically during fine-tuning

**ASPIRE Self-Evaluation**
- **Method**: Train model to evaluate its own answer correctness
- **Output Format**: Answer + confidence score or "I don't know" warning
- **Training**: Parameter-efficient fine-tuning (LoRA, soft prompts)

### 4.2 Limitations of Existing Approaches

1. **No Dedicated Token**: Most use natural language phrases ("I don't know", "I'm uncertain")
2. **Confidence-Based**: Rely on probability thresholds rather than learned abstention patterns
3. **Limited Gibberish Training**: No systematic training on synthetic nonsense data
4. **Post-Hoc Methods**: Often applied after training via prompting or thresholding

---

## 5. Hallucination Reduction Techniques

### 5.1 Grounding Methods

**Retrieval-Augmented Generation (RAG)**
- **Mechanism**: Ground generation in factual information from reliable sources
- **Effectiveness**: Vastly reduces hallucinations; doesn't rely solely on training data
- **Performance**: Semantic RAG + continual pre-training improves metrics by 40%
- **Key Paper**: "Ingest-And-Ground" (arXiv:2410.02825)

**Contextual Grounding with Guardrails**
- **Approach**: Check if responses are factually accurate based on provided sources
- **Implementation**: Modern guardrails with context-awareness

### 5.2 Training-Based Methods

**Direct Preference Optimization (DPO) for Factuality**
- **Approach**: Leverage automated fact-checking + preference learning
- **Performance**: >50% reduction in factual errors (LLaMA-2 7B)
- **Domains**: Particularly effective for biographies, medical questions

**Chain-of-Thought Prompting**
- **Mechanism**: Force model to articulate clear reasoning path
- **Effectiveness**: Reduces errors but not universally effective

### 5.3 Comprehensive Survey

**"A Comprehensive Survey of Hallucination Mitigation Techniques in LLMs"**
- **Citation**: arXiv:2401.01313
- **Key Insight**: No one-size-fits-all solution; optimal approach often mixes strategies
- **Factors**: Hallucinations arise from both prompt-dependent and model-intrinsic factors

---

## 6. Synthetic and Adversarial Data Generation

### 6.1 Synthetic Data for Edge Cases

**Purpose**:
- Include edge cases crucial for model performance but rare in real data
- Test models under various conditions
- Ensure reliable performance in unexpected scenarios

**Applications**:
- Enriching training sets with critical edge cases
- Testing ML models under controlled perturbations

### 6.2 Adversarial and Gibberish Research

**"Talking Nonsense: Probing LLMs' Understanding of Adversarial Gibberish Inputs"**
- **Citation**: arXiv:2404.17120
- **Key Finding**: "LM Babel" - nonsensical prompts that induce specific, coherent responses
- **Success Rate**: <3% for completely random strings
- **Insight**: Complexity of targeted text influences Babel prompt success

**Undertrained Tokens**
- **Issue**: Documents with glitch tokens used for tokenizer but removed from training
- **Consequence**: Anomalous behaviors, unexpected nonsensical generation
- **Lesson**: Need training coverage of all vocabulary to avoid tokenizer bias

### 6.3 GANs and Synthetic Data

**Generative Adversarial Networks**
- **Components**: Generator (creates synthetic data) + Discriminator (distinguishes real vs. fake)
- **Training**: Iterative improvement via discriminator feedback
- **NLP Application**: Text generation via transformers (GPT-like models)

**NVIDIA Nemotron-4**
- **Purpose**: Generate synthetic data for LLM training
- **Goal**: Create diverse data mimicking real-world characteristics
- **Outcome**: Improve data quality, increase performance and robustness

**Limitation in NLP**:
- Plain-vanilla GANs did not improve sentiment classifier performance
- Transformers (GPT) more effective for coherent text generation

---

## 7. Novelty of the UNKNOWN Token Sink Approach

### 7.1 What Already Exists

1. **Abstention via Confidence**: Thresholding token probabilities (ASPIRE, selective prediction)
2. **Refusal Tokens**: Safety-oriented tokens for value alignment
3. **Prompting**: "I don't know" phrases elicited through instructions
4. **Special Tokens**: [CLS], [SEP], [MASK] with specific training objectives
5. **OOD Detection**: Post-hoc methods using likelihood, perplexity, ensembles

### 7.2 What Is Novel

**1. Dedicated Epistemic Abstention Token**
   - Not for safety/values (refusal tokens)
   - Not generic special token ([CLS], [SEP])
   - Specifically for "I don't have knowledge for this input"

**2. Systematic Gibberish Training**
   - No prior work trains on synthetic nonsense data types:
     - Repetitive tokens
     - Random token sequences
     - Semantic nulls
     - Corrupted real data
   - Training explicitly associates gibberish patterns with abstention

**3. Pattern Recognition for Abstention**
   - Rather than confidence thresholding, model learns to recognize:
     - Semantic incoherence
     - Distributional anomalies
     - Meaninglessness patterns

**4. Hybrid Objective**
   - Normal language modeling on real data
   - Explicit abstention target on gibberish
   - Balances performance and calibration

**5. Proactive Rather Than Reactive**
   - Confidence thresholding is post-hoc (after generation)
   - UNKNOWN token can be emitted during generation when pattern recognized

### 7.3 Research Gap

The literature shows:
- **No work** explicitly combines a learnable abstention token with gibberish training
- **No systematic study** of training on multiple gibberish types (repetitive, random, null, corrupted)
- **No evaluation** of whether pattern-based abstention (vs. confidence-based) improves calibration

This represents a **genuine research contribution** at the intersection of:
- Selective prediction
- Special token training
- OOD detection
- Calibration
- Synthetic data generation

---

## 8. Potential Challenges and Failure Modes

### 8.1 Token Integration Challenges

**Embedding Initialization**:
- New tokens start with random embeddings
- Require sufficient training to develop meaningful representations
- Risk of underfitting if training insufficient

**Tokenization Issues**:
- Adding token changes tokenizer behavior
- Must ensure UNKNOWN doesn't break subword tokenization of adjacent tokens
- Need to save and distribute modified tokenizer with model

**Training Data Requirements**:
- Need balanced mixture of gibberish and real data
- Too much gibberish → model forgets language
- Too little gibberish → model doesn't learn UNKNOWN usage

### 8.2 Over-Abstention Risk

**Problem**: Model becomes too conservative, uses UNKNOWN excessively

**Manifestations**:
- High UNKNOWN rate on in-distribution data (>5-10%)
- Model abstains on answerable questions
- Reduced utility for end users

**Causes**:
- Gibberish ratio too high during training
- UNKNOWN token loss weight too strong
- Model pattern-matches superficial features (e.g., all long inputs)

**Mitigation**:
- Careful hyperparameter tuning (gibberish ratio, loss weighting)
- In-distribution accuracy as constraint during training
- Adversarial evaluation (hard but answerable questions)

### 8.3 Under-Abstention Risk

**Problem**: Model doesn't generalize beyond training gibberish types

**Manifestations**:
- Low UNKNOWN rate on held-out gibberish
- Low UNKNOWN rate on real OOD data
- Continued hallucination on novel nonsense

**Causes**:
- Insufficient gibberish diversity during training
- Model memorizes specific gibberish patterns without abstracting
- Real OOD data too different from synthetic gibberish

**Mitigation**:
- Diverse gibberish generation (multiple types and parameterizations)
- Evaluation on multiple OOD benchmarks
- Curriculum learning (start specific, increase diversity)

### 8.4 Multi-Token Output Handling

**Challenge**: Most LLM outputs are multi-token sequences

**Questions**:
- When to emit UNKNOWN? (Beginning, end, only output?)
- How to handle partial generation before UNKNOWN?
- Does UNKNOWN terminate generation or continue?

**Options**:
1. **Pure Abstention**: Output only "<UNKNOWN>\n" and stop
2. **Qualified Response**: "I'm uncertain. [attempt]<UNKNOWN>"
3. **Token-Level**: Mark uncertain spans with UNKNOWN
4. **Confidence-Gated**: Emit UNKNOWN if confidence below threshold at any step

**Trade-offs**:
- Pure abstention: Clear but all-or-nothing
- Qualified response: More nuanced but complicates training
- Token-level: Fine-grained but complex evaluation
- Confidence-gated: Hybrid approach, requires threshold tuning

### 8.5 Evaluation Challenges

**Defining "Appropriate" Abstention**:
- No ground truth for "correct" UNKNOWN rate on real OOD
- Different applications have different risk tolerances
- Hard to separate "truly unknowable" from "difficult but knowable"

**Metrics**:
- **Coverage-Accuracy Curves**: Standard but doesn't distinguish abstention reasons
- **Selective Risk**: Combines accuracy and coverage
- **AUACC/AUROC**: Area under curves, but still aggregate
- **Calibration Metrics**: ECE, Brier score for confidence-accuracy alignment

**Benchmarks**:
- Need diverse OOD datasets (semantic shift, domain shift, adversarial)
- Gibberish detection alone insufficient
- Must test on real challenging inputs

### 8.6 Training Dynamics

**Loss Balancing**:
- Standard language modeling loss on real data
- Cross-entropy loss for UNKNOWN on gibberish
- Risk of objective conflicts or one dominating

**Catastrophic Forgetting**:
- Adding gibberish training might degrade in-distribution performance
- Need continual learning or interleaved training strategies

**Curriculum Considerations**:
- Should gibberish be introduced gradually or from start?
- Does order of gibberish types matter?

### 8.7 Deployment and User Experience

**Interpretability**:
- Users need to understand what UNKNOWN means
- Is it "I don't know" or "this is nonsense" or "I'm uncertain"?

**Frequency Calibration**:
- Too many UNKNOWNs frustrate users
- Too few defeats purpose
- Application-dependent tuning needed

**Prompt Sensitivity**:
- Will instruction-tuned models override UNKNOWN with "always answer" prompts?
- How to ensure UNKNOWN mechanism is robust to user instructions?

---

## 9. Key Design Questions to Resolve

Based on the literature review, these critical design decisions must be made:

### 9.1 Token Addition
- **Q**: How to add UNKNOWN to tokenizer vocabulary?
- **Options**: `add_tokens()` vs. `add_special_tokens()`
- **Considerations**: Special token vs. regular token affects tokenization behavior

### 9.2 Gibberish Definition
- **Q**: What counts as "gibberish" for training?
- **Proposed Types**: Repetitive, random, semantic null, corrupted
- **Open**: Ratios, parameterizations (corruption %, repetition length, etc.)

### 9.3 Training Strategy
- **Q**: Fine-tune or train from scratch?
- **Fine-tuning**: More practical, tests on established models, faster
- **From scratch**: More control, avoids biases, resource-intensive
- **Recommendation**: Start with fine-tuning for feasibility study

### 9.4 Evaluation Framework
- **Q**: How to evaluate appropriate UNKNOWN usage?
- **Proposed**: In-dist accuracy (maintain), gibberish detection (>90%), real OOD (TBD)
- **Open**: Which real OOD benchmarks? How to measure hallucination reduction?

### 9.5 Output Format
- **Q**: Should UNKNOWN be single token output or part of a phrase?
- **Options**:
  - Pure token: "<UNKNOWN>"
  - Natural phrase: "I don't know about this."
  - Hybrid: "<UNKNOWN> I don't have information about this."
- **Trade-off**: Token-only is cleanest for training; phrase is more user-friendly

### 9.6 Loss Function
- **Q**: How to weight gibberish vs. normal data?
- **Options**: Equal weighting, ratio-based (1:10, 1:5), dynamic weighting
- **Open**: What prevents over/under-abstention?

---

## 10. Related Approaches and Baselines

For experimental validation, we should compare against:

### 10.1 Baselines

1. **Unmodified Model**: No UNKNOWN token, standard generation
2. **Confidence Thresholding**: Abstain if max token probability < threshold
3. **Prompt-Based Abstention**: "Answer 'I don't know' if unsure" instruction
4. **ASPIRE-style**: Self-evaluation with confidence scoring
5. **Conformal Prediction**: Prediction sets with coverage guarantees

### 10.2 Why Our Approach May Improve Over Baselines

**vs. Confidence Thresholding**:
- Confidence may be uncalibrated; model can be overconfident on gibberish
- Threshold is global; doesn't adapt to input type
- UNKNOWN learns patterns beyond raw probability

**vs. Prompt-Based**:
- Prompting is brittle and can be overridden
- Doesn't change model's internal representations
- UNKNOWN is trained into weights, more robust

**vs. ASPIRE**:
- ASPIRE uses confidence scoring, not pattern recognition
- Self-evaluation adds complexity
- UNKNOWN is simpler, single-pass generation

**vs. Conformal Prediction**:
- CP is post-hoc, doesn't change generation
- CP prediction sets can be unwieldy for text
- UNKNOWN is integrated into generation process

---

## 11. Recommended Literature for Deep Dive

### Must-Read Papers

1. **Know Your Limits** (arXiv:2407.18418) - Comprehensive abstention survey
2. **ASPIRE** (Google Research, EMNLP 2023) - Closest existing approach
3. **Calibrated LM Fine-Tuning** (EMNLP 2020) - Calibration issues in fine-tuning
4. **OOD Detection Survey** (TMLR 2023) - OOD methods for NLP
5. **Conformal Prediction Survey** (TACL 2024) - Uncertainty quantification with guarantees
6. **Refusal Tokens** (arXiv:2412.06748) - Most similar token-based approach

### Useful Background

7. **The Art of Abstention** (ACL 2021) - Selective prediction fundamentals
8. **LM-Polygraph** (TACL 2024) - UQ benchmarking
9. **Hallucination Survey** (arXiv:2401.01313) - Mitigation techniques
10. **Talking Nonsense** (arXiv:2404.17120) - LM Babel and gibberish

---

## 12. Conclusion

The proposed UNKNOWN token sink approach sits at a promising intersection of:
- Selective prediction (abstention when uncertain)
- Special token training (learned token semantics)
- OOD detection (recognizing distributional shifts)
- Calibration (aligning confidence with correctness)

**Key Insight**: While each component has been studied, **no prior work combines a learnable abstention token with systematic gibberish training**. This represents a novel contribution that could improve LLM reliability and safety.

**Primary Advantage**: Pattern-based abstention (recognizing gibberish) may be more robust than confidence-based methods (thresholding probabilities), especially when models are miscalibrated.

**Main Risk**: Over/under-abstention if training dynamics aren't carefully balanced. Success requires extensive empirical tuning and evaluation.

**Next Steps**: Design concrete implementation, generate gibberish data, select base model, and define evaluation protocol.
