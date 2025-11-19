# Model Selection and Dataset Recommendations

## Sub-1B Parameter Model Research Summary

### BERT: NOT RECOMMENDED ❌

**Critical Issue**: BERT is an **encoder-only** model designed for understanding tasks (classification, NER, QA), NOT text generation.

**Why BERT Won't Work**:
- Bidirectional attention makes it unsuitable for autoregressive generation
- No causal masking → cannot generate text token-by-token
- Designed for: classification, embeddings, token-level predictions
- Our task requires: sequence generation with explicit abstention

**Verdict**: BERT is fundamentally the wrong architecture for this research.

---

## Top Sub-1B Parameter Models (Ranked)

### 1. Gemma 3 270M ⭐ RECOMMENDED

**Specifications**:
- Parameters: 270M (170M embeddings + 100M transformer)
- Vocabulary: 256k tokens (handles rare/specific tokens well)
- Training: 6 trillion tokens (3x more than Gemma 1B!)
- Context: 32k tokens
- Architecture: Decoder-only (suitable for generation)
- Release: November 2025 (MOST RECENT)

**Advantages**:
- ✅ Latest SOTA for sub-1B models
- ✅ Exceptionally large vocabulary (good for tokenization experiments)
- ✅ Massive training data despite small size
- ✅ Optimized for on-device inference (energy efficient)
- ✅ Strong instruction-following capabilities pre-trained
- ✅ Available on Hugging Face with full support

**Considerations**:
- Large vocabulary means embedding layer dominates parameters
- Very new (Nov 2025) - less research/documentation than older models

**Verdict**: **Best choice for cutting-edge SOTA sub-1B model**

---

### 2. SmolLM-360M ⭐ STRONG ALTERNATIVE

**Specifications**:
- Parameters: 360M
- Training: 600B tokens on high-quality Cosmo-Corpus
- Context: Standard (likely 2048-4096)
- Architecture: Decoder-only with Grouped-Query Attention (GQA)
- Release: July 2024

**Advantages**:
- ✅ Outperforms models in its size category
- ✅ Trained on curated high-quality data (FineWeb-Edu, Cosmopedia v2, Python-Edu)
- ✅ GQA for faster inference
- ✅ Both base and instruction-tuned versions available
- ✅ Well-documented by Hugging Face
- ✅ Strong performance on commonsense reasoning

**Considerations**:
- Smaller than Gemma 3 270M in total parameters but well-optimized

**Verdict**: **Excellent balance of performance and efficiency**

---

### 3. Pythia-410M - RESEARCH-FRIENDLY OPTION

**Specifications**:
- Parameters: 410M
- Training: The Pile dataset
- Checkpoints: 154 intermediate checkpoints available
- Architecture: Decoder-only (GPT-style)
- Release: 2023 (EleutherAI)

**Advantages**:
- ✅ Deliberately designed for interpretability research
- ✅ 154 checkpoints allow studying training dynamics
- ✅ Deduped and non-deduped versions available
- ✅ All model sizes trained on same data in same order (scientific rigor)
- ✅ Extensive research community usage
- ✅ Well-understood baseline

**Considerations**:
- Older than Gemma/SmolLM (may have lower raw performance)
- Trained on The Pile (good but not latest curation)

**Verdict**: **Best for reproducible research with checkpoint analysis**

---

### 4. Qwen2-0.5B - PERFORMANCE LEADER (BUT SLIGHTLY LARGER)

**Specifications**:
- Parameters: 500M
- Context: 128k tokens (exceptional!)
- Multilingual: 29 languages
- Architecture: Decoder-only
- Release: 2024 (Alibaba Cloud)

**Advantages**:
- ✅ Best performing sub-1B model in benchmarks
- ✅ Massive 128k context window
- ✅ Instruction-tuned version excels at dialogue
- ✅ Multilingual capabilities

**Considerations**:
- ⚠️ 500M parameters (exceeds strict sub-500M threshold but under 1B)
- May require slightly more resources than 270M/360M models

**Verdict**: **If you can handle 500M, this has best raw performance**

---

### 5. SmolLM-135M - ULTRA-LIGHTWEIGHT OPTION

**Specifications**:
- Parameters: 135M
- Training: 600B tokens on Cosmo-Corpus
- Architecture: Decoder-only with GQA
- Release: July 2024

**Advantages**:
- ✅ Smallest viable option (fastest training/inference)
- ✅ Outperforms MobileLM-125M despite less training data
- ✅ Same high-quality training as SmolLM-360M
- ✅ Perfect for rapid prototyping

**Considerations**:
- Lower capacity may struggle with complex patterns
- Trade-off: speed vs. capability

**Verdict**: **Best for quick experiments with minimal resources**

---

## Final Model Recommendation

### Primary Choice: **Gemma 3 270M**

**Rationale**:
1. **Latest SOTA**: Most recent model (Nov 2025) with cutting-edge training
2. **Optimal Size**: 270M parameters fits resource constraints perfectly
3. **Massive Training**: 6 trillion tokens gives strong foundation
4. **Large Vocabulary**: 256k tokens ideal for adding custom UNKNOWN token
5. **Generation-Ready**: Decoder-only architecture, designed for text generation
6. **Well-Supported**: Google backing, Hugging Face integration, active development

### Backup Choice: **SmolLM-360M**

**Rationale**:
- If Gemma 3 270M has issues (too new, limited docs, compatibility)
- Proven track record (July 2024, well-tested)
- Excellent training data curation (FineWeb-Edu)
- Strong commonsense reasoning (important for gibberish detection)

---

## In-Distribution Training Dataset Recommendation

### Primary Choice: **FineWeb-Edu** ⭐

**Specifications**:
- Source: Hugging Face (HuggingFaceFW/fineweb-edu)
- Size: Subset of 1.3T tokens (can sample as needed)
- Quality: Educational content filtered from FineWeb
- Curation: Llama-3-70B-Instruct annotations for quality

**Advantages**:
- ✅ **10x more efficient**: Matches MMLU performance with 10x fewer tokens than C4
- ✅ **Educational focus**: High-quality, coherent text (good baseline for gibberish contrast)
- ✅ **SOTA quality**: Outperforms C4, Wikipedia, Dolma on benchmarks
- ✅ **Well-structured**: Clean, grammatically correct (easier to detect corruption)
- ✅ **Free & accessible**: Fully available on Hugging Face
- ✅ **Used by SmolLM**: Proven success with similar-sized models

**Why Better Than Alternatives**:
- **vs. Wikipedia**: More diverse, larger scale, educational but varied domains
- **vs. C4**: Higher quality, better educational benchmark performance
- **vs. OpenWebText**: More curated, less noise

**Verdict**: **FineWeb-Edu is the gold standard for small model fine-tuning in 2024-2025**

### Secondary Choice: **Wikipedia + FineWeb-Edu Mix**

**Rationale**:
- Wikipedia: Factual, well-structured, familiar baseline
- FineWeb-Edu: Educational diversity
- Mix ratio: 30% Wikipedia, 70% FineWeb-Edu
- Benefits: Combines structure with diversity

---

## Out-of-Distribution (OOD) Evaluation Dataset Recommendations

### 1. SQuAD 2.0 ⭐ PRIMARY OOD BENCHMARK

**Specifications**:
- Answerable: 100k questions from SQuAD 1.1
- Unanswerable: 50k adversarially written questions
- Domain: Reading comprehension

**Why Perfect for Our Research**:
- ✅ **Explicit abstention target**: Unanswerable questions = should output UNKNOWN
- ✅ **Adversarial design**: Unanswerable questions crafted to look answerable
- ✅ **Plausible answers present**: Tests if model recognizes semantic impossibility
- ✅ **Gold standard**: Widely used, trusted benchmark
- ✅ **Clear evaluation**: Precision/recall on abstention vs. answering

**Use Case**:
- Evaluate UNKNOWN usage on unanswerable vs. answerable questions
- Expected: High UNKNOWN rate on unanswerable, low on answerable
- Metric: Abstention accuracy (correctly abstaining when unanswerable)

**Verdict**: **Best OOD dataset for testing learned abstention**

---

### 2. Domain-Shifted Datasets - DISTRIBUTION SHIFT TESTING

**Options**:

**Medical → General (or vice versa)**:
- Train on FineWeb-Edu (general educational)
- Evaluate on PubMedQA or MedQA
- Tests: Does model abstain on unfamiliar medical terminology?

**Legal Domain**:
- Evaluate on LegalBench or CaseHOLD
- Tests: Abstention on specialized legal language

**Code**:
- Evaluate on HumanEval or MBPP (code generation)
- Tests: Does model recognize code as out-of-distribution if trained on prose?

**Rationale**:
- Real-world OOD (not synthetic gibberish)
- Tests generalization: Does gibberish training → domain shift abstention?
- Practical relevance: Models should know their domain limits

**Verdict**: **Include 1-2 domain-shifted datasets to test generalization**

---

### 3. Adversarial NLI - SEMANTIC COHERENCE TESTING

**Option**: Adversarial NLI (ANLI) or HANS

**Why Useful**:
- Contains adversarially constructed examples
- Tests semantic understanding vs. surface patterns
- Some examples are nonsensical or contradictory

**Use Case**:
- Evaluate if model abstains on contradictory premises
- Tests semantic null detection (similar to gibberish Type 3)

**Verdict**: **Optional, adds depth to semantic coherence evaluation**

---

### 4. TruthfulQA - HALLUCINATION BASELINE

**Specifications**:
- 817 questions spanning 38 categories
- Designed to test if models hallucinate false information
- Many questions have misleading common misconceptions

**Why Relevant**:
- Tests if UNKNOWN reduces hallucination on tricky questions
- Baseline: Standard model vs. UNKNOWN-augmented model
- Metric: Compare hallucination rate reduction

**Verdict**: **Include for hallucination reduction claims**

---

## Recommended OOD Evaluation Suite

### Comprehensive Evaluation Plan:

1. **Synthetic Gibberish** (held-out from training):
   - Type 1: Repetitive tokens
   - Type 2: Random sequences
   - Type 3: Semantic nulls
   - Type 4: Corrupted text
   - **Target**: >90% UNKNOWN usage

2. **SQuAD 2.0** (unanswerable questions):
   - Unanswerable: High UNKNOWN rate
   - Answerable: Low UNKNOWN rate
   - **Target**: >70% UNKNOWN on unanswerable, <10% on answerable

3. **Domain Shift** (choose 1-2):
   - Medical: PubMedQA
   - Code: HumanEval
   - **Target**: Moderate UNKNOWN rate (20-40%?)

4. **TruthfulQA**:
   - Compare hallucination rate: baseline vs. UNKNOWN model
   - **Target**: Measurable reduction in false claims

---

## Dataset Summary Table

| Dataset | Purpose | Size | Rationale |
|---------|---------|------|-----------|
| **FineWeb-Edu** | In-dist training | 1.3T tokens (sample) | SOTA quality, educational, 10x efficiency |
| **Synthetic Gibberish** | Training + OOD eval | Generated | Core innovation: train abstention |
| **SQuAD 2.0** | OOD eval | 150k questions | Unanswerable = explicit abstention target |
| **PubMedQA** | OOD eval | 1k QA pairs | Domain shift testing |
| **TruthfulQA** | Hallucination eval | 817 questions | Measure reduction in false claims |

---

## Addressing Your Questions

### Q1: Limited resources - BERT or Gemma 270M?
**A**: ❌ **NOT BERT** (wrong architecture). ✅ **Use Gemma 3 270M** (decoder-only, SOTA, generation-capable).

### Q2: Is Gemma 270M the latest SOTA sub-1B?
**A**: ✅ **YES**. Released November 2025, trained on 6T tokens, 256k vocab. Alternatives: SmolLM-360M (July 2024), Qwen2-0.5B (500M, best performance but larger).

### Q3: In-distribution dataset recommendation?
**A**: ✅ **FineWeb-Edu**. Outperforms C4, Wikipedia, Dolma. 10x more efficient. Used by SOTA small models like SmolLM. Educational content provides good contrast to gibberish.

### Q4: OOD evaluation datasets?
**A**: ✅ **SQuAD 2.0** (primary: unanswerable questions = perfect abstention test) + **PubMedQA** (domain shift) + **TruthfulQA** (hallucination reduction). Plus synthetic gibberish (held-out).

### Q5: Output behavior when UNKNOWN triggered?
**A**: ✅ **Stop immediately with generic response**. Simplest training objective. Clean evaluation. Could be: `<UNKNOWN>` or `<UNKNOWN>\nI don't have sufficient information to answer this.`

### Q6: 15% gibberish ratio - do you agree?
**A**: ⚠️ **15% is aggressive but could work**. Here's my analysis:

**Concerns**:
- Most selective prediction work uses confidence thresholds post-hoc (no explicit training)
- No prior work establishes optimal gibberish ratio
- Risk of over-abstention if too high

**Considerations**:
- 15% means 1 in 6-7 examples is gibberish during training
- Model sees significant abstention examples
- Higher than typical class imbalance in classification

**Recommendation**:
- **Start with 10%**, monitor in-dist performance closely
- If <5% UNKNOWN on in-dist and >90% on gibberish → working well
- If in-dist performance degrades OR over-abstention → reduce to 5%
- If under-abstention on gibberish → increase to 15%
- Make it a **tunable hyperparameter** with ablation study

**Modified Answer**: Start conservative (10%), prepared to adjust to 15% if needed.

### Q7: Acceptable in-dist accuracy degradation?
**A**: ✅ **<5% is excellent target**. Demonstrates UNKNOWN doesn't hurt normal performance. This is strict but achievable with careful loss weighting.

---

## Final Recommendation Summary

### Model: **Gemma 3 270M**
- Latest SOTA (Nov 2025)
- 270M parameters (fits resource constraints)
- 6T training tokens, 256k vocab
- Decoder-only (generation-ready)

### In-Distribution Data: **FineWeb-Edu**
- SOTA quality for small models
- Educational content (good gibberish contrast)
- Proven with SmolLM success

### Gibberish Ratio: **10% (start), tune to 15% if needed**
- Conservative start
- Monitor for over/under-abstention
- Ablation study: 5%, 10%, 15%, 20%

### OOD Evaluation:
1. Synthetic gibberish (held-out) - >90% UNKNOWN target
2. SQuAD 2.0 unanswerable - >70% UNKNOWN target
3. PubMedQA domain shift - moderate UNKNOWN
4. TruthfulQA - hallucination reduction

### Training Target:
- In-dist accuracy degradation: **<5%**
- Gibberish detection: **>90% UNKNOWN**
- Real OOD: **Appropriate abstention** (TBD per dataset)

---

**Status**: Ready to proceed to implementation design phase. ✅
