# Implementation Design

## Overview

This document specifies the complete implementation architecture for training the UNKNOWN token sink mechanism. It covers model modifications, training objectives, pipeline design, and evaluation framework.

---

## 1. Model Architecture Modifications

### 1.1 Adding the UNKNOWN Token

**Approach**: Add `<UNKNOWN>` as a special token to the tokenizer vocabulary.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model and tokenizer
model_name = "google/gemma-3-270m"  # Primary choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add UNKNOWN as special token
special_tokens_dict = {'additional_special_tokens': ['<UNKNOWN>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Resize model embeddings to accommodate new token
model.resize_token_embeddings(len(tokenizer))

# Get UNKNOWN token ID for later use
UNKNOWN_TOKEN_ID = tokenizer.convert_tokens_to_ids('<UNKNOWN>')
```

**Key Points**:
- Use `add_special_tokens()` to ensure `<UNKNOWN>` is treated as atomic (won't be split by BPE)
- New token embedding initialized randomly (will learn during training)
- Token ID stored for use in training/evaluation logic

### 1.2 Model Configuration

**No other architectural changes needed**:
- Use Gemma 3 270M as-is (decoder-only transformer)
- No additional layers or mechanisms
- UNKNOWN token learned purely through training data

**Rationale**: Simplicity. The token embedding and model weights will learn to associate gibberish patterns with UNKNOWN output through standard training.

---

## 2. Training Objective and Loss Function

### 2.1 Training Data Format

**Real Data Examples**:
```json
{
  "text": "The history of ancient Rome is fascinating.",
  "label": "real",
  "target": "standard_lm"
}
```

**Gibberish Data Examples**:
```json
{
  "text": "apple apple apple apple apple",
  "label": "gibberish",
  "target": "<UNKNOWN>"
}
```

### 2.2 Sequence Construction

**For Real Data** (Standard Language Modeling):
- **Input sequence**: `tokenizer.encode(text)`
- **Labels**: Same as input, shifted by 1 (standard causal LM)
- **Loss**: Cross-entropy on all positions

**For Gibberish Data** (UNKNOWN Target):
- **Input sequence**: `tokenizer.encode(gibberish_text + " <UNKNOWN>")`
- **Labels**: Same as input, BUT:
  - Mask gibberish token positions with `-100` (ignore in loss)
  - Only compute loss on `<UNKNOWN>` token position
- **Loss**: Cross-entropy only on UNKNOWN token

**Example**:
```python
# Gibberish text: "apple apple apple apple"
text = "apple apple apple apple"
target_seq = text + " <UNKNOWN>"

input_ids = tokenizer.encode(target_seq, return_tensors="pt")
# input_ids: [token_apple, token_apple, token_apple, token_apple, UNKNOWN_TOKEN_ID]

labels = input_ids.clone()
# Mask all positions except UNKNOWN
labels[0, :-1] = -100  # Ignore gibberish tokens in loss
# labels: [-100, -100, -100, -100, UNKNOWN_TOKEN_ID]

# Model learns: after seeing gibberish, predict UNKNOWN
```

### 2.3 Loss Function

**Combined Loss**:
```python
def compute_loss(model, batch):
    """
    Compute loss for mixed batch of real and gibberish data.

    Args:
        model: The language model
        batch: Dictionary with 'input_ids' and 'labels'

    Returns:
        loss: Scalar tensor
    """
    outputs = model(input_ids=batch['input_ids'],
                    labels=batch['labels'])

    loss = outputs.loss  # Standard cross-entropy with -100 masking
    return loss
```

**No special weighting needed**: Because gibberish examples naturally have fewer tokens contributing to loss (only UNKNOWN position), the loss is already balanced.

**Optional: Loss Weighting** (if over-abstention occurs):
```python
# Separate losses for real vs gibberish
real_loss = compute_loss(model, real_batch)
gibberish_loss = compute_loss(model, gibberish_batch)

# Weighted combination
alpha = 0.3  # Reduce gibberish loss weight (0.1-0.5 range)
total_loss = real_loss + alpha * gibberish_loss
```

**Initial Strategy**: Use standard mixed batches without special weighting. Add weighting only if over-abstention detected.

---

## 3. Training Pipeline

### 3.1 Data Loading

```python
from datasets import load_dataset, concatenate_datasets

# Load FineWeb-Edu
fineweb = load_dataset("HuggingFaceFW/fineweb-edu",
                       name="sample-10BT",  # 10B token sample
                       split="train")

# Sample 90K examples for training
real_train = fineweb.shuffle(seed=42).select(range(90_000))

# Load generated gibberish (from GIBBERISH_GENERATION.md)
gibberish_train = load_dataset("json",
                               data_files="data/train/gibberish_data.jsonl",
                               split="train")

# Combine datasets
train_dataset = concatenate_datasets([real_train, gibberish_train])
train_dataset = train_dataset.shuffle(seed=42)  # Mix together
```

### 3.2 Data Preprocessing

```python
def preprocess_function(examples, tokenizer, max_length=512):
    """
    Tokenize and prepare examples for training.

    For real data: standard tokenization
    For gibberish: add UNKNOWN target and mask appropriately
    """
    input_ids_list = []
    labels_list = []

    for text, label in zip(examples['text'], examples['label']):
        if label == "real":
            # Standard LM: predict next token
            tokens = tokenizer.encode(text,
                                      max_length=max_length,
                                      truncation=True)
            input_ids_list.append(tokens)
            labels_list.append(tokens.copy())  # Standard causal LM

        elif label == "gibberish":
            # UNKNOWN target: append <UNKNOWN> and mask gibberish tokens
            target_text = text + " <UNKNOWN>"
            tokens = tokenizer.encode(target_text,
                                      max_length=max_length,
                                      truncation=True)

            # Create labels: mask everything except UNKNOWN position
            labels = [-100] * len(tokens)
            # Find UNKNOWN token position (should be last)
            unknown_id = tokenizer.convert_tokens_to_ids('<UNKNOWN>')
            for i, token_id in enumerate(tokens):
                if token_id == unknown_id:
                    labels[i] = unknown_id  # Only compute loss here

            input_ids_list.append(tokens)
            labels_list.append(labels)

    return {
        'input_ids': input_ids_list,
        'labels': labels_list
    }

# Apply preprocessing
tokenized_train = train_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names
)
```

### 3.3 Training Loop

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output/unknown_token_sink",
    overwrite_output_dir=True,

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch size: 32

    # Learning rate
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Logging
    logging_dir="./logs",
    logging_steps=100,
    report_to="tensorboard",

    # Optimization
    fp16=True,  # Mixed precision training
    dataloader_num_workers=4,

    # Reproducibility
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)

trainer.train()
```

---

## 4. Hyperparameters

### 4.1 Model Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Gemma 3 270M | Latest SOTA sub-1B, 6T tokens training |
| Vocabulary Size | 256k + 1 (`<UNKNOWN>`) | Large vocab + our token |
| Max Sequence Length | 512 tokens | Balance memory and context |

### 4.2 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 32 (16 x 2 accum) | Fits in single GPU with gradient accumulation |
| Learning Rate | 5e-5 | Standard fine-tuning rate for small models |
| LR Schedule | Cosine with warmup | Smooth learning, 10% warmup |
| Epochs | 3 | Sufficient for fine-tuning, avoid overfitting |
| Optimizer | AdamW | Standard for transformers |
| Weight Decay | 0.01 | Light regularization |
| Gradient Clipping | 1.0 | Prevent instability |

### 4.3 Data Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training Size | 100K examples | 90K real + 10K gibberish |
| Gibberish Ratio | 10% (initial) | Conservative start, tune to 15% if needed |
| Validation Size | 11K examples | 10K real + 1K gibberish |
| Max Seq Length | 512 tokens | Standard for small models |

### 4.4 Tunable Hyperparameters

**If under-abstention (gibberish detection < 90%)**:
- Increase gibberish ratio: 10% → 15%
- Increase loss weight on gibberish: α = 0.5 → 1.0
- Train longer: 3 → 5 epochs
- Reduce learning rate: 5e-5 → 2e-5 (more careful learning)

**If over-abstention (in-dist UNKNOWN > 5%)**:
- Decrease gibberish ratio: 10% → 5%
- Decrease loss weight on gibberish: α = 1.0 → 0.3
- Add regularization: Penalize UNKNOWN on in-dist validation
- Early stopping based on in-dist UNKNOWN rate

---

## 5. Evaluation Framework

### 5.1 Metrics

**In-Distribution (FineWeb-Edu)**:
- **Perplexity**: Should remain close to baseline (within 10%)
- **UNKNOWN Rate**: Percentage of examples where model outputs UNKNOWN (target: <5%)
- **Accuracy**: Downstream task performance if applicable

**Synthetic Gibberish (Held-Out)**:
- **UNKNOWN Rate**: Percentage detecting gibberish (target: >90%)
- **Breakdown by Type**: Separate rates for Types 1-4

**SQuAD 2.0**:
- **Answerable UNKNOWN Rate**: Should be low (target: <10%)
- **Unanswerable UNKNOWN Rate**: Should be high (target: >70%)
- **F1 Score** (answerable): Standard QA metric

**Domain Shift (PubMedQA)**:
- **UNKNOWN Rate**: Moderate (target: 20-40%)

**Hallucination (TruthfulQA)**:
- **Truthful Response Rate**: Compare baseline vs UNKNOWN model
- **Hallucination Rate**: False claims per response

### 5.2 Evaluation Implementation

```python
def evaluate_unknown_rate(model, tokenizer, dataset, unknown_token_id):
    """
    Compute UNKNOWN usage rate on evaluation dataset.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        unknown_token_id: ID of UNKNOWN token

    Returns:
        unknown_rate: Percentage of examples with UNKNOWN output
    """
    model.eval()
    unknown_count = 0
    total = 0

    for example in dataset:
        input_text = example['text']
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id
            )

        # Check if UNKNOWN in generated tokens
        generated_ids = outputs[0, input_ids.shape[1]:]  # Only new tokens
        if unknown_token_id in generated_ids:
            unknown_count += 1

        total += 1

    unknown_rate = (unknown_count / total) * 100
    return unknown_rate
```

### 5.3 Baseline Comparison

**Baselines to implement**:

1. **Unmodified Gemma 3 270M**
   - No UNKNOWN token, standard generation
   - Measure: hallucination rate, perplexity on OOD

2. **Confidence Thresholding**
   - Use max token probability as confidence
   - Abstain if confidence < threshold (tune threshold)
   - Measure: coverage-accuracy trade-off

3. **Prompt-Based Abstention**
   - Add instruction: "Answer the question. If you don't know, say 'I don't know'."
   - Measure: abstention rate and accuracy

**Comparison Metrics**:
- **Coverage-Accuracy Curves**: Plot accuracy vs coverage at different thresholds
- **AUACC**: Area under accuracy-coverage curve
- **Selective Risk**: Error rate on non-abstained examples

---

## 6. Code Structure

```
unknown_token_sink/
├── src/
│   ├── data/
│   │   ├── generate_gibberish.py      # Implement 4 gibberish types
│   │   ├── load_fineweb.py            # Load and sample FineWeb-Edu
│   │   └── preprocessing.py           # Tokenization and formatting
│   ├── models/
│   │   ├── load_model.py              # Load Gemma, add UNKNOWN token
│   │   └── modeling_utils.py          # Helper functions
│   ├── training/
│   │   ├── train.py                   # Main training script
│   │   ├── loss.py                    # Loss computation (if custom weighting)
│   │   └── callbacks.py               # Custom callbacks for monitoring
│   ├── evaluation/
│   │   ├── evaluate_indist.py         # In-distribution metrics
│   │   ├── evaluate_gibberish.py      # Synthetic gibberish detection
│   │   ├── evaluate_squad.py          # SQuAD 2.0 evaluation
│   │   ├── evaluate_domain_shift.py   # PubMedQA evaluation
│   │   └── evaluate_hallucination.py  # TruthfulQA evaluation
│   ├── baselines/
│   │   ├── confidence_threshold.py    # Baseline 1
│   │   └── prompt_based.py            # Baseline 2
│   └── utils/
│       ├── logging.py                 # Experiment tracking
│       └── metrics.py                 # Metric computation
├── configs/
│   ├── model_config.yaml              # Model parameters
│   ├── training_config.yaml           # Training hyperparameters
│   └── eval_config.yaml               # Evaluation settings
├── scripts/
│   ├── generate_all_data.sh           # Run all data generation
│   ├── run_training.sh                # Run training with logging
│   └── run_evaluation.sh              # Run full evaluation suite
├── notebooks/
│   ├── 01_explore_data.ipynb          # Inspect generated data
│   ├── 02_analyze_results.ipynb       # Analyze evaluation results
│   └── 03_visualize_metrics.ipynb     # Plot metrics and comparisons
├── data/                               # Data directory (see GIBBERISH_GENERATION.md)
├── output/                             # Model checkpoints
├── logs/                               # TensorBoard logs
├── results/                            # Evaluation results
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # uv project file
└── README.md                           # Setup and usage instructions
```

---

## 7. Experiment Tracking

### 7.1 Logging

**TensorBoard Metrics** (logged every 100 steps):
- Training loss
- Validation loss (real data)
- Validation loss (gibberish data)
- Learning rate
- Gradient norm

**Custom Metrics** (logged every 500 steps):
- In-dist UNKNOWN rate (validation)
- Gibberish detection rate (validation)
- Perplexity on in-dist validation

### 7.2 Checkpointing

**Save Strategy**:
- Save checkpoint every 500 steps
- Keep only best 3 checkpoints (by validation loss)
- Save final model at end

**Checkpoint Contents**:
- Model weights
- Tokenizer (with UNKNOWN token)
- Optimizer state
- Training arguments
- Random seeds

### 7.3 Reproducibility

**Random Seeds**:
- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- CUDA: `torch.cuda.manual_seed_all(42)`
- Hugging Face: `set_seed(42)`

**Deterministic Operations**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Environment**:
- Log library versions (transformers, torch, etc.)
- Log GPU type and count
- Log exact command-line arguments

---

## 8. Training Procedure

### 8.1 Phase 1: Initial Training (10% Gibberish)

1. Generate all data (see GIBBERISH_GENERATION.md)
2. Add UNKNOWN token to Gemma 3 270M
3. Train for 3 epochs with 10% gibberish ratio
4. Evaluate on all benchmarks
5. Analyze results:
   - If gibberish detection >90% AND in-dist UNKNOWN <5%: **SUCCESS** ✓
   - If gibberish detection <90%: Proceed to Phase 2
   - If in-dist UNKNOWN >5%: Proceed to Phase 3

### 8.2 Phase 2: Increase Gibberish (If Under-Abstention)

1. Increase gibberish ratio to 15%
2. Regenerate training mixture
3. Train from Phase 1 best checkpoint (or restart)
4. Evaluate and compare

### 8.3 Phase 3: Decrease Gibberish (If Over-Abstention)

1. Decrease gibberish ratio to 5%
2. Regenerate training mixture
3. Add UNKNOWN penalty on in-dist validation
4. Train from Phase 1 best checkpoint (or restart)
5. Evaluate and compare

### 8.4 Phase 4: Hyperparameter Tuning

Once basic mechanism works:
- Ablate gibberish types (train with only 1-2 types, test generalization)
- Ablate mixing ratios
- Try different learning rates
- Try different model sizes (SmolLM-360M backup)

---

## 9. Deployment Considerations

### 9.1 Inference

```python
def generate_with_unknown(model, tokenizer, input_text, max_length=100):
    """
    Generate response with UNKNOWN detection.

    Returns:
        response: Generated text
        abstained: Boolean indicating if model abstained
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    unknown_id = tokenizer.convert_tokens_to_ids('<UNKNOWN>')

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Check for UNKNOWN
    abstained = '<UNKNOWN>' in generated_text or unknown_id in generated_ids

    if abstained:
        return "<UNKNOWN>", True
    else:
        return generated_text, False
```

### 9.2 Post-Processing

**When UNKNOWN detected**:
- Option 1: Return empty response with "Unable to respond" message
- Option 2: Fallback to retrieval (RAG)
- Option 3: Route to human review

**User-facing message**:
```
"I don't have sufficient information to provide a reliable response to this input."
```

---

## 10. Success Criteria Checklist

- [ ] **Model trains successfully** without errors or divergence
- [ ] **In-dist perplexity** within 10% of baseline Gemma 3 270M
- [ ] **In-dist UNKNOWN rate** < 5%
- [ ] **Synthetic gibberish detection** > 90% across all 4 types
- [ ] **SQuAD 2.0 unanswerable detection** > 70%
- [ ] **SQuAD 2.0 answerable UNKNOWN** < 10%
- [ ] **TruthfulQA hallucination reduction** measurable (>5% improvement)
- [ ] **Baselines outperformed** on coverage-accuracy trade-off

**Minimum Viable Success**: First 4 criteria (train successfully, in-dist maintained, gibberish detected)

**Full Success**: All 8 criteria met

---

## 11. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Over-abstention | Medium | High | Start with 10% gibberish, monitor in-dist UNKNOWN closely |
| Under-abstention | Medium | Medium | Prepared to increase to 15%, add loss weighting |
| Catastrophic forgetting | Low | High | Use large real:gibberish ratio (90:10), validate frequently |
| Training instability | Low | Medium | Use gradient clipping, warmup, conservative LR |
| Poor generalization to real OOD | High | Medium | Diverse gibberish types, test on multiple OOD datasets |
| Gemma 3 270M incompatibility | Low | High | Have SmolLM-360M as backup model |

---

**Status**: Implementation design complete. Ready for IMPLEMENTATION.md with phased execution plan. ✅
