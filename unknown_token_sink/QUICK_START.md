# UNKNOWN Token Sink - Quick Start Guide

Get started with training and evaluating the UNKNOWN Token Sink model in minutes.

## Prerequisites

- Python 3.9+
- NVIDIA GPU with 16GB+ VRAM (recommended) or CPU (slow)
- ~20GB disk space for data and models

## Step 1: Environment Setup (5 minutes)

```bash
cd unknown_token_sink

# Initialize environment and install dependencies
bash scripts/setup_environment.sh

# Activate environment
source .venv/bin/activate

# Validate setup
python scripts/validate_setup.py
```

**Expected output:** All 7 checks should pass ✓

## Step 2: Generate Training Data (30-60 minutes)

```bash
# Generate all data (FineWeb-Edu + synthetic gibberish)
bash scripts/generate_all_data.sh
```

This will:
- Download 105K FineWeb-Edu examples (90K train, 10K val, 5K test)
- Generate 10K synthetic gibberish examples (4 types)
- Create mixed training dataset (90% real, 10% gibberish)
- Prepare validation and test sets

**Data location:**
- `data/train/mixed_training_data.jsonl` - Training set (~100K examples)
- `data/validation/validation_data.jsonl` - Validation set (~10K examples)
- `data/test/test_real.jsonl` - In-distribution test set (~5K examples)
- `data/test/gibberish_synthetic.jsonl` - Gibberish test set (~1K examples)

## Step 3: Train Model (4-8 hours on GPU, days on CPU)

```bash
# Train with default configuration
python src/training/train.py \
    --config configs/training_config.yaml \
    --data_dir ./data \
    --output_dir ./output

# Optional: Launch with TensorBoard
python src/training/train.py \
    --config configs/training_config.yaml \
    --data_dir ./data \
    --output_dir ./output \
    --tensorboard
```

**Training progress:**
- Checkpoints saved to `output/checkpoint-*`
- Final model saved to `output/final_model`
- Logs available in `output/logs`

**Monitor with TensorBoard:**
```bash
tensorboard --logdir output/logs
```

## Step 4: Evaluate Model (10-20 minutes)

```bash
# Run comprehensive evaluation
bash scripts/run_evaluation.sh
```

**Evaluation metrics:**
- UNKNOWN rate on in-distribution data (target: <5%)
- UNKNOWN rate on gibberish (target: >90%)
- Perplexity on in-distribution data
- Classification F1 score
- Confidence metrics

**Results saved to:**
- `results/evaluation_results.json` - Detailed metrics
- Console output - Summary and success criteria

## Step 5: Inspect Results

```bash
# View formatted results
cat results/evaluation_results.json | python -m json.tool

# Check success criteria
grep "SUCCESS CRITERIA" -A 20 <evaluation_output>
```

## Expected Results

### Success Criteria

**In-Distribution Data (FineWeb-Edu Test):**
- ✓ UNKNOWN rate: <5%
- ✓ Perplexity degradation: <10% vs baseline
- ✓ Next-token accuracy maintained

**Synthetic Gibberish:**
- ✓ UNKNOWN rate: >90%
- ✓ High precision (few false positives)
- ✓ High recall (few false negatives)

**Classification Metrics:**
- ✓ F1 score: >0.85
- ✓ Accuracy: >0.90

## Quick Configuration Changes

### Adjust Gibberish Ratio

Edit `configs/training_config.yaml`:
```yaml
data:
  gibberish_ratio: 0.15  # Change from 0.10 to 0.15 (15%)
```

Then regenerate data:
```bash
bash scripts/generate_all_data.sh
```

### Change Model

Edit `configs/model_config.yaml`:
```yaml
model:
  name: "HuggingFaceTB/SmolLM-360M"  # Use SmolLM instead of Gemma
```

### Adjust Training

Edit `configs/training_config.yaml`:
```yaml
training:
  num_train_epochs: 5  # Train longer
  learning_rate: 3.0e-5  # Lower learning rate
  per_device_train_batch_size: 8  # Smaller batches for limited VRAM
```

## Common Issues

### Out of Memory (OOM)

**Solution:** Reduce batch size in `configs/training_config.yaml`
```yaml
training:
  per_device_train_batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

### Data Download Fails

**Solution:** The script automatically falls back to CC-100 if FineWeb-Edu is unavailable.

### Model Not Loading

**Solution:** Check internet connection. The script will try SmolLM-360M as backup.

### CPU Training Too Slow

**Solution:**
1. Reduce dataset size in data generation
2. Use fewer epochs (1-2 instead of 3)
3. Consider using Google Colab or cloud GPU

## Advanced Usage

### Custom Gibberish Generation

```python
from src.data.generate_gibberish import GibberishGenerator

generator = GibberishGenerator()
gibberish = generator.generate_all(
    num_examples=1000,
    type_distribution={
        'repetitive': 0.5,  # More repetitive
        'random': 0.2,
        'semantic_null': 0.2,
        'corrupted': 0.1
    }
)
```

### Manual Evaluation

```python
from src.models.unknown_token_model import UnknownTokenModel

# Load trained model
model = UnknownTokenModel()
model.load("./output/final_model")

# Test on custom text
text = "apple apple apple apple"
prediction = model.generate(text, max_new_tokens=10)
print(prediction)  # Should contain <UNKNOWN>
```

### Export Model for Deployment

```python
from src.models.unknown_token_model import UnknownTokenModel

model = UnknownTokenModel()
model.load("./output/final_model")

# Save in HuggingFace format
model.model.save_pretrained("./deployed_model")
model.tokenizer.save_pretrained("./deployed_model")
```

## Next Steps

1. **Experiment with configurations** - Try different gibberish ratios, model sizes
2. **Evaluate on real OOD data** - Test on domain-shifted datasets
3. **Analyze failure cases** - Examine examples where model fails to abstain
4. **Compare baselines** - Implement confidence thresholding for comparison
5. **Scale up** - Try larger models if resources available

## File Structure Reference

```
unknown_token_sink/
├── configs/               # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── eval_config.yaml
├── data/                  # Generated data (created by scripts)
│   ├── train/
│   ├── validation/
│   └── test/
├── output/                # Training outputs (created during training)
│   ├── checkpoint-*/
│   └── final_model/
├── results/               # Evaluation results (created during eval)
│   └── evaluation_results.json
├── src/                   # Source code
│   ├── data/             # Data generation and loading
│   ├── models/           # Model architecture
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utilities
├── scripts/              # Helper scripts
│   ├── setup_environment.sh
│   ├── validate_setup.py
│   ├── generate_all_data.sh
│   └── run_evaluation.sh
└── README.md             # Main documentation
```

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `IMPLEMENTATION.md` for technical details
3. Examine `RESEARCH.md` for background research
4. Check GitHub issues (if applicable)

## Citation

If you use this code for research, please cite:
```
@software{unknown_token_sink,
  title={UNKNOWN Token Sink: In-Training-Distribution Knowledge Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ai_research}
}
```
