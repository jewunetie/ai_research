# Implementation Roadmap

## Overview

This document provides a phased, step-by-step implementation plan for the UNKNOWN Token Sink research prototype. Each phase includes specific deliverables, validation steps, and success criteria.

**Estimated Total Time**: 2-3 weeks (assuming single GPU, part-time work)

---

## Prerequisites

### Environment Setup

```bash
# Navigate to project directory
cd /home/user/ai_research/unknown_token_sink

# Initialize uv project
uv init
uv venv
source .venv/bin/activate

# Install dependencies
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu118
uv add transformers datasets accelerate
uv add tensorboard wandb  # Experiment tracking
uv add pandas numpy scipy scikit-learn
uv add jupyter ipywidgets  # For notebooks
```

### Hardware Requirements

- **Minimum**: 1x GPU with 16GB VRAM (e.g., RTX 4060 Ti, T4)
- **Recommended**: 1x GPU with 24GB+ VRAM (e.g., RTX 3090, A10)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space (datasets + models + checkpoints)

### Directory Structure

```bash
mkdir -p src/{data,models,training,evaluation,baselines,utils}
mkdir -p configs scripts notebooks
mkdir -p data/{train,validation,test}
mkdir -p output logs results
```

---

## Phase 0: Project Setup (Day 1)

### Objectives
- Set up development environment
- Create project structure
- Verify GPU access and library installations

### Tasks

**0.1 Environment Validation**
```bash
# Test PyTorch and GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Test Transformers
python -c "from transformers import AutoTokenizer; print('Transformers installed successfully')"
```

**0.2 Create Configuration Files**

`configs/model_config.yaml`:
```yaml
model:
  name: "google/gemma-3-270m"
  max_length: 512
  unknown_token: "<UNKNOWN>"

tokenizer:
  padding_side: "right"
  truncation: true
```

`configs/training_config.yaml`:
```yaml
training:
  output_dir: "./output/unknown_token_sink"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: true
  seed: 42

data:
  real_data_size: 90000
  gibberish_data_size: 10000
  gibberish_ratio: 0.10
  max_length: 512
```

**0.3 Create README.md**

Document:
- Project overview
- Setup instructions
- Running training
- Running evaluation

### Deliverables
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] GPU access verified
- [ ] Project structure created
- [ ] Configuration files created
- [ ] README.md written

### Validation
```bash
# Run setup validation script
python scripts/validate_setup.py
```

**Success Criteria**: All imports work, GPU detected, project structure exists

---

## Phase 1: Data Generation (Days 2-3)

### Objectives
- Implement 4 gibberish generation types
- Sample FineWeb-Edu data
- Create training/validation/test splits
- Validate data quality

### Tasks

**1.1 Implement Gibberish Generators**

`src/data/generate_gibberish.py`:
- Implement `generate_repetitive()` - Type 1
- Implement `generate_random_sequences()` - Type 2
- Implement `generate_semantic_nulls()` - Type 3
- Implement `generate_corrupted_data()` - Type 4
- Implement `generate_all_gibberish()` - Main function

**1.2 Load and Sample FineWeb-Edu**

`src/data/load_fineweb.py`:
```python
from datasets import load_dataset

def load_fineweb_edu(split_sizes={'train': 90000, 'val': 10000, 'test': 5000}):
    """Load and sample FineWeb-Edu dataset."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train"
    )

    dataset = dataset.shuffle(seed=42)

    # Split dataset
    train = dataset.select(range(split_sizes['train']))
    val = dataset.select(range(split_sizes['train'],
                                split_sizes['train'] + split_sizes['val']))
    test = dataset.select(range(split_sizes['train'] + split_sizes['val'],
                                 split_sizes['train'] + split_sizes['val'] + split_sizes['test']))

    return {'train': train, 'val': val, 'test': test}
```

**1.3 Generate All Datasets**

`scripts/generate_all_data.sh`:
```bash
#!/bin/bash

echo "Generating gibberish data..."
python src/data/generate_gibberish.py \
    --output_dir data \
    --train_size 8000 \
    --val_size 1000 \
    --test_size 1000 \
    --seed 42

echo "Loading FineWeb-Edu..."
python src/data/load_fineweb.py \
    --output_dir data \
    --train_size 90000 \
    --val_size 10000 \
    --test_size 5000 \
    --seed 42

echo "Data generation complete!"
```

**1.4 Data Quality Validation**

Create `notebooks/01_explore_data.ipynb`:
- Load generated gibberish samples
- Visualize distribution of types, lengths, parameters
- Manual inspection of 50 random samples
- Check for accidental coherence in Type 2
- Verify FineWeb-Edu samples look reasonable

### Deliverables
- [ ] All 4 gibberish generators implemented
- [ ] FineWeb-Edu loader implemented
- [ ] Training data generated (90K real + 8K gibberish)
- [ ] Validation data generated (10K real + 1K gibberish)
- [ ] Test data generated (5K real + 1K gibberish)
- [ ] Data exploration notebook completed
- [ ] Data quality validated

### Validation
```bash
# Check data files exist
ls -lh data/train/
ls -lh data/validation/
ls -lh data/test/

# Run data validation
python src/data/validate_data.py
```

**Success Criteria**:
- All data files present with correct sizes
- Gibberish types evenly distributed (25% each)
- No obviously malformed examples
- FineWeb-Edu samples are coherent text

**Estimated Time**: 1-2 days

---

## Phase 2: Model Setup and Token Integration (Day 4)

### Objectives
- Load Gemma 3 270M model
- Add UNKNOWN token to vocabulary
- Verify token integration
- Save modified model and tokenizer

### Tasks

**2.1 Implement Model Loading**

`src/models/load_model.py`:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_with_unknown_token(model_name="google/gemma-3-270m"):
    """
    Load model and add UNKNOWN token.

    Returns:
        model: Modified model
        tokenizer: Modified tokenizer
        unknown_token_id: ID of UNKNOWN token
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Add UNKNOWN token
    special_tokens_dict = {'additional_special_tokens': ['<UNKNOWN>']}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added} special tokens")
    print(f"New vocab size: {len(tokenizer)}")

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Get UNKNOWN token ID
    unknown_token_id = tokenizer.convert_tokens_to_ids('<UNKNOWN>')
    print(f"UNKNOWN token ID: {unknown_token_id}")

    return model, tokenizer, unknown_token_id
```

**2.2 Test Token Integration**

`scripts/test_token_integration.py`:
```python
from src.models.load_model import load_model_with_unknown_token

# Load model
model, tokenizer, unknown_id = load_model_with_unknown_token()

# Test tokenization
test_texts = [
    "This is normal text.",
    "apple apple apple",
    "Text with <UNKNOWN> token"
]

for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Contains UNKNOWN: {unknown_id in tokens}")
    print("---")

# Test generation (before training - should be random)
input_text = "apple apple apple"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)
generated = tokenizer.decode(outputs[0])
print(f"Generation test: {generated}")
```

**2.3 Save Modified Model**

```python
# Save model with UNKNOWN token
model.save_pretrained("./output/gemma_270m_with_unknown")
tokenizer.save_pretrained("./output/gemma_270m_with_unknown")
```

### Deliverables
- [ ] Model loading script implemented
- [ ] UNKNOWN token successfully added
- [ ] Token integration tested
- [ ] Modified model and tokenizer saved

### Validation
```bash
# Run token integration test
python scripts/test_token_integration.py
```

**Success Criteria**:
- Model loads without errors
- Tokenizer vocabulary size increases by 1
- UNKNOWN token correctly tokenized and decoded
- Model can generate (output will be random before training)

**Estimated Time**: 0.5-1 day

---

## Phase 3: Data Preprocessing and Pipeline (Day 5)

### Objectives
- Implement data preprocessing for training
- Create DataLoader with proper batching
- Handle mixed real/gibberish examples
- Verify loss computation works correctly

### Tasks

**3.1 Implement Preprocessing**

`src/data/preprocessing.py` - See IMPLEMENTATION_DESIGN.md section 3.2 for full code

Key functions:
- `preprocess_function()`: Tokenize and create labels
- `DataCollatorWithPadding`: Custom collator for mixed data

**3.2 Create Data Pipeline**

`src/training/data_pipeline.py`:
```python
from datasets import load_from_disk, concatenate_datasets
from src.data.preprocessing import preprocess_function

def create_dataloaders(tokenizer, config):
    """Create train and validation dataloaders."""

    # Load real data
    real_train = load_from_disk("data/train/real_data")
    real_val = load_from_disk("data/validation/real_data")

    # Load gibberish data
    gib_train = load_from_disk("data/train/gibberish_data")
    gib_val = load_from_disk("data/validation/gibberish_data")

    # Combine and shuffle
    train_dataset = concatenate_datasets([real_train, gib_train])
    train_dataset = train_dataset.shuffle(seed=42)

    val_dataset = concatenate_datasets([real_val, gib_val])

    # Tokenize
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    return train_tokenized, val_tokenized
```

**3.3 Test Data Pipeline**

`scripts/test_data_pipeline.py`:
- Load one batch
- Print input_ids, labels, attention_mask
- Verify labels are correct (-100 for gibberish positions, token IDs for others)
- Compute loss on single batch (should work without errors)

### Deliverables
- [ ] Preprocessing functions implemented
- [ ] Data pipeline created
- [ ] Dataloaders tested
- [ ] Single-batch loss computation verified

### Validation
```bash
python scripts/test_data_pipeline.py
```

**Success Criteria**:
- Dataloaders return batches correctly
- Labels properly formatted (-100 masking works)
- Loss computation runs without errors
- Real and gibberish examples correctly processed

**Estimated Time**: 1 day

---

## Phase 4: Training (Days 6-8)

### Objectives
- Train model with 10% gibberish ratio
- Monitor training metrics
- Validate on held-out data
- Save best checkpoint

### Tasks

**4.1 Implement Training Script**

`src/training/train.py` - See IMPLEMENTATION_DESIGN.md section 3.3 for full code

**4.2 Run Training**

```bash
# Start training
python src/training/train.py \
    --config configs/training_config.yaml \
    --model_config configs/model_config.yaml \
    --output_dir output/unknown_token_sink \
    --run_name "unknown_token_10pct_gibberish"

# Monitor training
tensorboard --logdir logs/
```

**4.3 Monitor Training Metrics**

Watch for:
- Training loss decreases smoothly
- Validation loss decreases
- No divergence or NaN losses
- In-dist UNKNOWN rate stays low (<5%)
- Gibberish detection rate increases

**4.4 Handle Issues**

If training diverges:
- Reduce learning rate (5e-5 → 2e-5)
- Increase warmup (0.1 → 0.2)
- Check for data issues

If in-dist UNKNOWN rate increases:
- Reduce gibberish ratio (10% → 5%)
- Add regularization on in-dist validation

### Deliverables
- [ ] Training script implemented
- [ ] Training completed for 3 epochs
- [ ] TensorBoard logs saved
- [ ] Best model checkpoint saved
- [ ] Training curves look reasonable

### Validation

Check TensorBoard:
- Loss curves decrease
- No sudden spikes or divergence
- Validation metrics improve

**Success Criteria**:
- Training completes without errors
- Final training loss < 3.0
- Validation loss < 3.5
- Model checkpoint saved successfully

**Estimated Time**: 2-3 days (including training time)

---

## Phase 5: Evaluation (Days 9-11)

### Objectives
- Evaluate on all benchmarks
- Measure UNKNOWN rates
- Compare to baselines
- Analyze results

### Tasks

**5.1 In-Distribution Evaluation**

`src/evaluation/evaluate_indist.py`:
- Load FineWeb-Edu test set (5K examples)
- Compute perplexity
- Measure UNKNOWN rate
- Compare to baseline Gemma 3 270M

**5.2 Synthetic Gibberish Evaluation**

`src/evaluation/evaluate_gibberish.py`:
- Load held-out gibberish test set (1K examples)
- Measure UNKNOWN rate overall
- Break down by type (Type 1-4)
- Analyze failure cases

**5.3 SQuAD 2.0 Evaluation**

`src/evaluation/evaluate_squad.py`:
- Load SQuAD 2.0 dataset
- Separate answerable vs unanswerable
- Measure UNKNOWN rates for both
- Compute F1 score on answerable (standard metric)

**5.4 Domain Shift Evaluation**

`src/evaluation/evaluate_domain_shift.py`:
- Load PubMedQA dataset
- Measure UNKNOWN rate
- Analyze patterns (what triggers UNKNOWN?)

**5.5 Hallucination Evaluation**

`src/evaluation/evaluate_hallucination.py`:
- Load TruthfulQA dataset
- Generate responses with baseline and UNKNOWN model
- Compare truthfulness rates
- Measure hallucination reduction

**5.6 Baseline Comparisons**

Implement and evaluate:
- Unmodified Gemma 3 270M
- Confidence thresholding baseline
- Prompt-based abstention baseline

**5.7 Run Full Evaluation Suite**

```bash
bash scripts/run_evaluation.sh
```

This script runs all evaluation scripts and saves results to `results/`

### Deliverables
- [ ] In-distribution metrics computed
- [ ] Gibberish detection rates measured
- [ ] SQuAD 2.0 results obtained
- [ ] Domain shift evaluation completed
- [ ] Hallucination rates compared
- [ ] Baseline comparisons done
- [ ] Results saved to JSON/CSV files

### Validation

Check results against targets:
- In-dist UNKNOWN < 5%: ✓ / ✗
- Gibberish detection > 90%: ✓ / ✗
- SQuAD unanswerable > 70%: ✓ / ✗
- SQuAD answerable < 10%: ✓ / ✗
- Hallucination reduction > 5%: ✓ / ✗

**Success Criteria**:
- All evaluation scripts run without errors
- Results saved in structured format
- At minimum: in-dist maintained + gibberish detected >90%

**Estimated Time**: 2-3 days

---

## Phase 6: Analysis and Iteration (Days 12-14)

### Objectives
- Analyze results deeply
- Identify failure modes
- Tune hyperparameters if needed
- Document findings

### Tasks

**6.1 Result Analysis**

Create `notebooks/02_analyze_results.ipynb`:
- Load all evaluation results
- Plot metrics across datasets
- Compare to baselines
- Identify patterns in UNKNOWN usage

**6.2 Failure Analysis**

- Find examples where model should abstain but doesn't (false negatives)
- Find examples where model abstains incorrectly (false positives)
- Categorize failure modes
- Propose fixes

**6.3 Hyperparameter Tuning (If Needed)**

**If gibberish detection < 90%**:
- Increase gibberish ratio to 15%
- Retrain (return to Phase 4)
- Re-evaluate (Phase 5)

**If in-dist UNKNOWN > 5%**:
- Decrease gibberish ratio to 5%
- Add regularization
- Retrain and re-evaluate

**6.4 Ablation Studies (Optional)**

- Train with only one gibberish type, test generalization
- Vary gibberish ratios (5%, 10%, 15%, 20%)
- Try different loss weightings
- Document impact of each choice

**6.5 Create Visualizations**

`notebooks/03_visualize_metrics.ipynb`:
- Coverage-accuracy curves
- UNKNOWN rate by dataset
- Comparison to baselines (bar charts)
- Failure case examples

### Deliverables
- [ ] Result analysis notebook completed
- [ ] Failure modes identified and documented
- [ ] Hyperparameter tuning completed (if needed)
- [ ] Ablation studies done (optional)
- [ ] Visualizations created
- [ ] Findings documented

### Validation

Review with fresh eyes:
- Do results make sense?
- Are failure modes understandable?
- Did tuning improve metrics?

**Success Criteria**:
- Clear understanding of model behavior
- Documented failure modes
- Tuned hyperparameters (if needed)
- Visualizations tell clear story

**Estimated Time**: 2-3 days

---

## Phase 7: Documentation and Wrap-Up (Day 15)

### Objectives
- Document final results
- Update README
- Create demo notebook
- Prepare for sharing/presentation

### Tasks

**7.1 Update Documentation**

- Update README with final results
- Document how to run training and evaluation
- Add example usage
- Note any caveats or limitations

**7.2 Create Demo Notebook**

`notebooks/04_demo.ipynb`:
- Load trained model
- Interactive: user inputs text, model generates with UNKNOWN detection
- Show examples of successful abstention
- Show examples of normal generation

**7.3 Create Results Summary**

`RESULTS.md`:
- Final metrics table
- Comparison to baselines
- Key findings
- Limitations
- Future work

**7.4 Code Cleanup**

- Remove unused code
- Add docstrings to all functions
- Format code consistently
- Add type hints

**7.5 Final Validation**

Run everything end-to-end:
```bash
bash scripts/generate_all_data.sh
bash scripts/run_training.sh
bash scripts/run_evaluation.sh
```

Ensure reproducibility.

### Deliverables
- [ ] README updated with complete instructions
- [ ] Demo notebook created
- [ ] RESULTS.md written
- [ ] Code cleaned and documented
- [ ] Final validation passed

### Validation

- Can someone else run the code from README instructions?
- Do all scripts work without modification?
- Are results reproducible?

**Success Criteria**:
- Complete, clear documentation
- Runnable demo
- Reproducible pipeline

**Estimated Time**: 1 day

---

## Success Criteria Summary

### Minimum Viable Success ✅

- [ ] Training completes without errors
- [ ] In-distribution perplexity within 10% of baseline
- [ ] In-distribution UNKNOWN rate < 5%
- [ ] Synthetic gibberish detection > 90%

### Full Success 🎯

All above, plus:
- [ ] SQuAD 2.0 unanswerable detection > 70%
- [ ] SQuAD 2.0 answerable UNKNOWN < 10%
- [ ] TruthfulQA hallucination reduction > 5%
- [ ] Outperforms baselines on coverage-accuracy

### Stretch Goals 🚀

- [ ] Ablation studies completed
- [ ] Paper-ready results and visualizations
- [ ] Open-source release prepared

---

## Risk Mitigation Plan

| If This Happens | Then Do This |
|-----------------|--------------|
| GPU out of memory | Reduce batch size to 8, increase gradient accumulation to 4 |
| Training diverges | Reduce LR to 2e-5, increase warmup to 20% |
| In-dist UNKNOWN > 5% | Reduce gibberish ratio to 5%, add in-dist regularization |
| Gibberish detection < 90% | Increase gibberish ratio to 15%, increase training epochs to 5 |
| Gemma 3 270M download fails | Use SmolLM-360M as backup model |
| FineWeb-Edu access issues | Use Wikipedia or C4 dataset instead |
| SQuAD 2.0 results poor | Focus on synthetic gibberish success, note SQuAD as future work |

---

## Next Steps After Completion

1. **Paper Writing**: Document methodology and results for publication
2. **Open Source Release**: Clean up code, add license, publish to GitHub
3. **Extended Evaluation**: Test on more OOD datasets
4. **Ablation Studies**: Systematic analysis of design choices
5. **Larger Models**: Try Gemma 7B or Llama 3 8B
6. **Applications**: Deploy in real-world use cases (chatbots, QA systems)

---

## Quick Start Command Sequence

```bash
# Phase 0: Setup
cd /home/user/ai_research/unknown_token_sink
uv venv && source .venv/bin/activate
uv add torch transformers datasets accelerate tensorboard

# Phase 1: Data
bash scripts/generate_all_data.sh

# Phase 2: Model
python src/models/load_model.py

# Phase 3: Pipeline
python scripts/test_data_pipeline.py

# Phase 4: Train
python src/training/train.py

# Phase 5: Evaluate
bash scripts/run_evaluation.sh

# Phase 6: Analyze
jupyter notebook notebooks/02_analyze_results.ipynb

# Phase 7: Document
# Update README, create RESULTS.md
```

---

**Status**: Implementation roadmap complete. Ready for execution. ✅

**Total Estimated Time**: 2-3 weeks (part-time) or 1 week (full-time)

**Ready to proceed to Phase 0 and begin implementation!**
