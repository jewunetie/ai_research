# Experiments

This directory contains experiment configurations and runners for the LLM self-compression QA research.

## Directory Structure

```
experiments/
├── configs/                    # Experiment configurations
│   ├── pilot_config.yaml      # Pilot experiment (10 docs)
│   └── main_config.yaml       # Main experiment (100 docs)
│
└── main/                      # Main experiment scripts
    ├── run_main.py           # Experiment runner
    └── analyze_main.py       # Results analysis
```

## Quick Start

### 1. Pilot Experiment (10 documents)

Test the pipeline on a small scale:

```bash
python experiments/main/run_main.py --config experiments/configs/pilot_config.yaml
```

### 2. Main Experiment (100 documents)

Run the full experiment:

```bash
python experiments/main/run_main.py --config experiments/configs/main_config.yaml
```

### 3. Resume from Checkpoint

If an experiment is interrupted, resume from the last checkpoint:

```bash
python experiments/main/run_main.py --config experiments/configs/main_config.yaml --resume
```

### 4. Analyze Results

After the experiment completes:

```bash
python experiments/main/analyze_main.py results/main/main_*.json
```

## Configuration

Experiments are configured via YAML files. Key parameters:

### Experiment Settings
- `num_documents`: Number of documents to process
- `dataset`: Dataset to use (default: "cnn_dailymail")
- `questions_per_document`: Questions per document (default: 10)

### Model Settings
- `name`: Model to use (e.g., "gpt-5.1-chat-latest")
- `temperature`: Sampling temperature (0.0 = deterministic)
- `seed`: Random seed for reproducibility

### Compression Settings
- `token_limit`: Maximum tokens for compression (default: 1500)
- `variants`: List of compression prompts to test

### Baselines
- `full_context`: Upper bound (no compression)
- `no_context`: Lower bound (empty context)
- `random_tokens`: Random token selection

## Features

### Progress Tracking
- Real-time progress bars (requires `tqdm`)
- ETA estimation
- Success/failure tracking
- Live cost estimates

### Cost Tracking
- Per-API-call token counting
- Running cost estimates
- Final cost summary
- Model-specific pricing

### Checkpointing
- Automatic checkpoints every N documents
- Resume from last checkpoint with `--resume`
- No data loss on interruption

### Error Handling
- Graceful failure handling
- Failed documents tracked separately
- Continue on errors
- Detailed error logging

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "config": {...},
  "timestamp": "20251123_143022",
  "results": [
    {
      "doc_id": "cnn_dm_0",
      "original_tokens": 1234,
      "variants": {
        "self_compression": {
          "compression_metadata": {...},
          "answers": [...]
        }
      },
      "baselines": {
        "full_context": {
          "metadata": {...},
          "answers": [...]
        }
      }
    }
  ],
  "failed_documents": [...],
  "summary": {
    "total_attempted": 100,
    "successful": 98,
    "failed": 2
  },
  "cost_summary": {
    "calls_made": 2150,
    "estimated_cost": 45.67
  }
}
```

## Analysis

The analysis script (`analyze_main.py`) provides:

- Summary statistics by condition
- Statistical comparisons (paired t-tests)
- Effect sizes (Cohen's d)
- Detailed CSV export for further analysis
- Publication-ready tables

## Tips

1. **Start with pilot**: Always test with `pilot_config.yaml` first
2. **Monitor costs**: Check cost estimates during pilot before running main
3. **Use checkpoints**: Long experiments benefit from frequent checkpoints
4. **Verify API**: Run `python verify_api.py` before experiments
5. **Check pricing**: Update pricing in config with actual GPT-5.1 rates

## Troubleshooting

### "Module not found" errors
```bash
pip install -e .
```

### "No progress bars"
```bash
pip install tqdm
```

### "API rate limit exceeded"
Add retry logic or reduce parallelism

### "Out of memory"
Process fewer documents or reduce batch size

## Advanced Usage

### Custom Compression Prompts

Edit config to add new variants:

```yaml
compression:
  variants:
    - name: "my_custom_compression"
      description: "Custom compression strategy"
      prompt_key: "MY_CUSTOM_PROMPT"
```

Then add the prompt to `src/compression/prompts.py`.

### Different Models

Change the model in config:

```yaml
model:
  name: "gpt-4o"  # More cost-effective
```

### Subset Testing

Test on specific document ranges by modifying the loader in `run_main.py`.
