# Gibberish Data Generation Specification

## Overview

This document provides precise specifications for generating synthetic gibberish data to train the UNKNOWN token abstention mechanism. Each gibberish type is designed to test different aspects of distributional anomaly detection.

---

## Generation Parameters

### Global Parameters

- **Total Training Dataset Size**: 100K examples (adjustable based on compute)
  - 90% FineWeb-Edu (90K examples)
  - 10% Gibberish (10K examples) - **initial ratio, tune to 15% if needed**

- **Gibberish Type Distribution** (within 10K gibberish examples):
  - Type 1 (Repetitive): 25% (2,500 examples)
  - Type 2 (Random): 25% (2,500 examples)
  - Type 3 (Semantic Null): 25% (2,500 examples)
  - Type 4 (Corrupted): 25% (2,500 examples)

- **Data Splits**:
  - Training: 80% (8,000 gibberish examples)
  - Validation: 10% (1,000 gibberish examples)
  - Test (held-out): 10% (1,000 gibberish examples)

- **Target Output**: All gibberish inputs → `<UNKNOWN>` token only, then stop generation

---

## Type 1: Repetitive Token Sequences

### Purpose
Test if model recognizes unnatural repetition as anomalous.

### Generation Algorithm

```python
def generate_repetitive(vocab, num_examples):
    """
    Generate repetitive token sequences.

    Args:
        vocab: Tokenizer vocabulary
        num_examples: Number of examples to generate

    Returns:
        List of repetitive sequences
    """
    examples = []

    # Token pools
    common_words = ["the", "a", "an", "is", "are", "was", "and", "or", "but"]
    nouns = ["apple", "dog", "cat", "house", "car", "tree", "book", "table"]
    verbs = ["run", "jump", "eat", "sleep", "think", "write", "read"]
    adjectives = ["big", "small", "red", "blue", "fast", "slow", "happy"]

    all_pools = [common_words, nouns, verbs, adjectives]

    for _ in range(num_examples):
        # Select token to repeat
        pool = random.choice(all_pools)
        token = random.choice(pool)

        # Repetition count: 5-20
        repetitions = random.randint(5, 20)

        # Generate sequence
        sequence = " ".join([token] * repetitions)
        examples.append(sequence)

    return examples
```

### Parameter Ranges

- **Repetition Count**: 5-20 repetitions
  - 5 reps: Minimal to be detected as anomalous
  - 20 reps: Extreme repetition
  - Uniform distribution across range

- **Token Selection Strategy**:
  - 40% common function words (the, a, is, etc.)
  - 30% nouns (concrete objects)
  - 20% verbs (action words)
  - 10% adjectives (descriptors)

### Examples

```
"the the the the the the the"                    # 7 repetitions, common word
"apple apple apple apple apple"                  # 5 repetitions, noun
"run run run run run run run run run run"        # 10 repetitions, verb
"blue blue blue blue blue blue blue blue blue blue blue blue blue blue blue"  # 15 reps
```

### Quality Control

- Ensure no accidental creation of valid constructions like "very very very" (which can be emphatic)
- Avoid tokens that commonly repeat in natural text ("ha ha ha", "no no no")

---

## Type 2: Random Token Sequences

### Purpose
Test if model recognizes distributional incoherence - tokens that never co-occur naturally.

### Generation Algorithm

```python
def generate_random_sequences(tokenizer, num_examples):
    """
    Generate random token sequences with no coherence.

    Args:
        tokenizer: Hugging Face tokenizer
        num_examples: Number of examples to generate

    Returns:
        List of random sequences with coherence checking
    """
    examples = []
    vocab = list(tokenizer.get_vocab().keys())

    # Filter out special tokens
    vocab = [t for t in vocab if not t.startswith('<') and not t.startswith('[')]

    for _ in range(num_examples):
        # Sequence length: 10-50 tokens
        length = random.randint(10, 50)

        # Sample random tokens
        max_attempts = 10
        for attempt in range(max_attempts):
            tokens = random.sample(vocab, length)
            sequence = " ".join(tokens)

            # Coherence check: ensure perplexity is high
            # (Optional: can compute perplexity with small model)
            # If perplexity < threshold, resample

            # Simple heuristic: check for common n-grams
            if not contains_common_bigrams(sequence):
                examples.append(sequence)
                break
        else:
            # If all attempts failed, accept anyway (rare)
            examples.append(sequence)

    return examples

def contains_common_bigrams(text):
    """
    Check if text contains common English bigrams.
    Returns True if common bigrams found (reject this sample).
    """
    common_bigrams = [
        ("the", "of"), ("in", "the"), ("to", "the"), ("and", "the"),
        ("of", "a"), ("to", "a"), ("in", "a"), ("for", "the")
    ]

    words = text.lower().split()
    for i in range(len(words) - 1):
        if (words[i], words[i+1]) in common_bigrams:
            return True
    return False
```

### Parameter Ranges

- **Sequence Length**: 10-50 tokens
  - 10 tokens: Short gibberish (quick to identify)
  - 50 tokens: Long gibberish (harder pattern)
  - Distribution: Uniform random across range

- **Token Sampling**:
  - Uniform sampling from vocabulary (no frequency weighting)
  - Ensures rare and common tokens mixed
  - No replacement (no duplicate tokens in single sequence)

### Examples

```
"xylophone transistor umbrella galaxy 27 purple banana legislation microscope quantum"
"velvet hypothesis chimney fractal negotiate sprinkle membrane paradox asteroid 19"
"algorithm whisper turbulent magnify cucumber fossil neutron improvise catalyst rhythm"
```

### Quality Control

- **Coherence Detection**:
  - Check for common bigrams/trigrams (reject if found)
  - Optional: Compute perplexity with GPT-2 small; reject if < threshold
  - Ensures truly random, not accidentally coherent

- **Diversity**:
  - Track generated sequences, avoid near-duplicates
  - Ensure wide vocabulary coverage

---

## Type 3: Semantic Null Sequences

### Purpose
Test if model recognizes semantic contradictions and logical impossibilities.

### Generation Algorithm

```python
def generate_semantic_nulls(num_examples):
    """
    Generate grammatically correct but semantically meaningless sentences.

    Uses template-based generation with contradictory terms.
    """
    examples = []

    # Templates with contradiction slots
    templates = [
        "Colorless {color} {noun} {verb} {adverb}",
        "The {adj1} {noun} is {contradictory_adj}",
        "{adj1} {adj2} {noun} {verb} {contradictory_adverb}",
        "A {size1} {size2} {noun} {verb}",
        "The {state1} {noun} remains {contradictory_state}"
    ]

    # Contradictory word sets
    colors = ["green", "red", "blue", "yellow", "purple"]
    nouns = ["ideas", "thoughts", "concepts", "theories", "dreams"]
    verbs = ["sleep", "dance", "run", "whisper", "sing"]
    adverbs = ["furiously", "silently", "quickly", "loudly", "brightly"]

    # Contradictory pairs
    contradictions = {
        "frozen": "burning", "silent": "loud", "motionless": "racing",
        "empty": "full", "invisible": "bright", "dead": "alive",
        "ancient": "newborn", "tiny": "massive", "smooth": "rough"
    }

    size_contradictions = [
        ("tiny", "gigantic"), ("microscopic", "enormous"),
        ("small", "huge"), ("miniature", "colossal")
    ]

    for _ in range(num_examples):
        template = random.choice(templates)

        # Fill template with contradictory terms
        if "Colorless" in template:
            sentence = template.format(
                color=random.choice(colors),
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                adverb=random.choice(adverbs)
            )
        elif "{adj1} {adj2}" in template:
            adj1, adj2 = random.choice(list(contradictions.items()))
            sentence = template.format(
                adj1=adj1, adj2=adj2,
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                contradictory_adverb="quietly" if "loud" in adj2 else "loudly"
            )
        elif "{size1} {size2}" in template:
            size1, size2 = random.choice(size_contradictions)
            sentence = template.format(
                size1=size1, size2=size2,
                noun=random.choice(nouns),
                verb=random.choice(verbs)
            )
        else:
            # Other templates
            adj1 = random.choice(list(contradictions.keys()))
            sentence = template.format(
                adj1=adj1,
                noun=random.choice(nouns),
                contradictory_adj=contradictions[adj1],
                state1=adj1,
                contradictory_state=contradictions[adj1]
            )

        examples.append(sentence)

    return examples
```

### Templates and Examples

**Template 1: Colorless + Color**
```
"Colorless green ideas sleep furiously"
"Colorless red concepts dance silently"
"Colorless blue theories whisper loudly"
```

**Template 2: Direct Contradiction**
```
"The frozen fire burns brightly"
"The silent scream echoes loudly"
"The motionless race runs quickly"
```

**Template 3: Contradictory Adjectives**
```
"Invisible bright lights shine dimly"
"Dead alive creatures breathe slowly"
"Ancient newborn stars collapse eternally"
```

**Template 4: Size Contradictions**
```
"A tiny gigantic mouse squeaks"
"The microscopic enormous atom vibrates"
"A miniature colossal idea emerges"
```

**Template 5: State Contradictions**
```
"The empty box remains full"
"The smooth surface stays rough"
"The solid liquid flows rigidly"
```

### Parameter Ranges

- **Template Variety**: 10-15 distinct templates
- **Word Pool Sizes**:
  - Nouns: 20-30 words
  - Verbs: 20-30 words
  - Adjectives: 30-40 words
  - Contradictory pairs: 15-20 pairs

- **Sentence Length**: Typically 5-10 words (natural for templates)

### Quality Control

- **Grammatical Correctness**: All sentences must parse correctly
- **Semantic Impossibility**: Every sentence must contain logical contradiction
- **Diversity**: Track template usage to ensure balanced distribution

---

## Type 4: Corrupted Real Data

### Purpose
Test model robustness to partial input corruption and gradual coherence degradation.

### Generation Algorithm

```python
def generate_corrupted_data(original_texts, tokenizer, num_examples):
    """
    Corrupt real text by randomly replacing tokens.

    Args:
        original_texts: List of clean text samples from FineWeb-Edu
        tokenizer: Hugging Face tokenizer
        num_examples: Number of corrupted examples to generate

    Returns:
        List of corrupted texts with varying corruption rates
    """
    examples = []
    vocab = list(tokenizer.get_vocab().keys())
    vocab = [t for t in vocab if not t.startswith('<') and not t.startswith('[')]

    # Corruption rates
    corruption_rates = [0.10, 0.30, 0.50, 0.70]
    rate_distribution = [0.20, 0.30, 0.30, 0.20]  # More mid-range corruption

    for _ in range(num_examples):
        # Sample original text
        original = random.choice(original_texts)
        tokens = original.split()

        # Skip very short texts
        if len(tokens) < 10:
            continue

        # Select corruption rate
        corruption_rate = random.choices(corruption_rates, rate_distribution)[0]

        # Number of tokens to corrupt
        num_corrupt = int(len(tokens) * corruption_rate)

        # Select positions to corrupt
        corrupt_positions = random.sample(range(len(tokens)), num_corrupt)

        # Corrupt tokens
        corrupted_tokens = tokens.copy()
        for pos in corrupt_positions:
            corrupted_tokens[pos] = random.choice(vocab)

        corrupted_text = " ".join(corrupted_tokens)
        examples.append({
            "text": corrupted_text,
            "corruption_rate": corruption_rate,
            "original_length": len(tokens)
        })

    return examples
```

### Parameter Ranges

- **Corruption Rates**: 10%, 30%, 50%, 70%
  - **10%**: Subtle corruption, mostly readable
  - **30%**: Noticeable corruption, partially coherent
  - **50%**: Heavy corruption, mostly incoherent
  - **70%**: Extreme corruption, nearly random

- **Corruption Rate Distribution**:
  - 10%: 20% of corrupted examples
  - 30%: 30% of corrupted examples
  - 50%: 30% of corrupted examples
  - 70%: 20% of corrupted examples
  - (More weight on mid-range to test gradient)

- **Source Text Length**: 20-100 tokens (to ensure meaningful corruption)

### Examples

**Original Text**:
```
"The quick brown fox jumps over the lazy dog in the garden."
```

**10% Corruption** (1 token):
```
"The quick brown fox jumps over the lazy xylophone in the garden."
```

**30% Corruption** (3 tokens):
```
"The quantum brown fox jumps over the lazy dog transistor the garden."
```

**50% Corruption** (5 tokens):
```
"The quantum microscope fox jumps legislation the lazy dog transistor umbrella garden."
```

**70% Corruption** (7 tokens):
```
"Velvet quantum microscope algorithm jumps legislation hypothesis lazy dog transistor umbrella fossil."
```

### Quality Control

- **Position Randomness**: Ensure corrupted positions are truly random (not clustered)
- **Token Randomness**: Replacement tokens should be diverse, not repeated
- **Length Preservation**: Original text length should be maintained
- **Source Diversity**: Sample from diverse FineWeb-Edu documents

---

## Data Mixing Strategy

### Training Data Composition

Total: 100K examples

**Real Data (90K examples, 90%)**:
- Source: FineWeb-Edu
- Selection: Random sampling from dataset
- Preprocessing: Standard tokenization, no special handling
- Target: Standard next-token prediction

**Gibberish Data (10K examples, 10%)**:
- Type 1 Repetitive: 2,500 examples (25%)
- Type 2 Random: 2,500 examples (25%)
- Type 3 Semantic Null: 2,500 examples (25%)
- Type 4 Corrupted: 2,500 examples (25%)
- Target: Output `<UNKNOWN>` token and stop

### Interleaving Strategy

**Option A: Random Shuffle** (RECOMMENDED)
- Shuffle all 100K examples together
- Model sees gibberish and real data intermixed
- Prevents learning temporal patterns
- Mirrors realistic deployment (OOD inputs arrive unpredictably)

**Option B: Curriculum Learning** (Alternative)
- Epoch 1-2: 5% gibberish, 95% real
- Epoch 3-4: 10% gibberish, 90% real
- Epoch 5+: 15% gibberish, 85% real (if tuning up)
- Gradually increase difficulty
- May prevent catastrophic forgetting

**Recommendation**: Start with Option A (random shuffle). Use Option B if in-distribution performance degrades.

---

## Validation and Test Splits

### Validation Set (Monitoring During Training)

- **Real Data**: 10K examples from FineWeb-Edu (held-out)
- **Gibberish**: 1K examples (250 each type, held-out from training)
- **Purpose**:
  - Monitor in-dist perplexity
  - Monitor gibberish detection rate
  - Early stopping if over-abstention detected

### Test Set (Final Evaluation)

- **Synthetic Gibberish (Held-Out)**: 1K examples
  - Type 1: 250 examples
  - Type 2: 250 examples
  - Type 3: 250 examples
  - Type 4: 250 examples
  - Target: >90% UNKNOWN rate

- **Real In-Distribution**: 5K FineWeb-Edu examples
  - Target: <5% UNKNOWN rate
  - Target: Perplexity similar to baseline

- **SQuAD 2.0**: Official test set
  - Answerable: Target <10% UNKNOWN
  - Unanswerable: Target >70% UNKNOWN

- **PubMedQA**: 1K medical QA pairs
  - Target: Moderate UNKNOWN (20-40%)

- **TruthfulQA**: 817 questions
  - Target: Measure hallucination reduction

---

## Implementation Checklist

- [ ] Implement Type 1 generator (repetitive)
- [ ] Implement Type 2 generator (random sequences)
- [ ] Implement Type 3 generator (semantic nulls)
- [ ] Implement Type 4 generator (corrupted data)
- [ ] Create data mixture pipeline
- [ ] Generate training set (80K real + 8K gibberish)
- [ ] Generate validation set (10K real + 1K gibberish)
- [ ] Generate test set (5K real + 1K gibberish)
- [ ] Quality control: Manual inspection of samples
- [ ] Quality control: Automated coherence checking (Type 2)
- [ ] Save datasets in Hugging Face Datasets format
- [ ] Document random seeds for reproducibility

---

## Reproducibility

All generation scripts must:
1. Accept random seed parameter
2. Log generation parameters (corruption rates, repetition counts, etc.)
3. Save metadata with each example (type, parameters)
4. Version control generation code

**Recommended Seeds**:
- Training set: seed=42
- Validation set: seed=123
- Test set: seed=456

---

## Expected Outputs

### File Structure

```
data/
├── train/
│   ├── real_data.jsonl          # 80K FineWeb-Edu examples
│   └── gibberish_data.jsonl     # 8K gibberish examples (mixed types)
├── validation/
│   ├── real_data.jsonl          # 10K FineWeb-Edu examples
│   └── gibberish_data.jsonl     # 1K gibberish examples
├── test/
│   ├── real_data.jsonl          # 5K FineWeb-Edu examples
│   ├── gibberish_synthetic.jsonl # 1K held-out synthetic gibberish
│   ├── squad2_answerable.jsonl
│   ├── squad2_unanswerable.jsonl
│   ├── pubmedqa.jsonl
│   └── truthfulqa.jsonl
└── metadata.json                 # Generation parameters and statistics
```

### JSONL Format

Each line:
```json
{
  "text": "apple apple apple apple apple",
  "label": "gibberish",
  "type": "repetitive",
  "parameters": {"token": "apple", "repetitions": 5},
  "target": "<UNKNOWN>"
}
```

or

```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "label": "real",
  "source": "fineweb-edu",
  "target": "standard_lm"
}
```

---

## Tuning Strategy

### If Under-Abstention (Gibberish Detection < 90%)

1. **Increase gibberish ratio**: 10% → 15%
2. **Increase gibberish diversity**: Generate more examples per type
3. **Adjust loss weighting**: Increase UNKNOWN loss weight
4. **Curriculum learning**: Start with extreme gibberish, add subtler cases

### If Over-Abstention (In-Dist UNKNOWN > 5%)

1. **Decrease gibberish ratio**: 10% → 5%
2. **Reduce UNKNOWN loss weight**: Lower penalty on gibberish
3. **Add regularization**: Penalize UNKNOWN on in-dist validation
4. **Quality check gibberish**: Ensure no accidentally coherent samples

### If Domain Shift Abstention Too High

1. **Add domain-diverse training data**: Include medical/legal/code samples in training
2. **Reduce gibberish extremity**: Focus on subtle anomalies
3. **Adjust evaluation expectations**: Real OOD may legitimately warrant abstention

---

**Status**: Gibberish generation specification complete. Ready for implementation. ✅
