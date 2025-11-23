# Phase 6: Full Evaluation Plan

**Goal**: Run complete evaluation on 50 problems for both baseline and MECE conditions

**Date**: 2025-11-23
**Status**: READY TO START

---

## Prerequisites Checklist

Before running full evaluation, ensure all preparation steps are complete:

- [ ] Dependencies installed (`uv sync`)
- [ ] Model loading tested
- [ ] Inference tested on sample problems
- [ ] All metrics (ME, CE, Accuracy) working
- [ ] End-to-end pipeline validated on 3 test problems
- [ ] No errors in test results

**Run preparation script:**
```bash
# Step 1: Install dependencies
./scripts/setup_dependencies.sh

# Step 2: Test full pipeline
python scripts/test_full_pipeline.py
```

---

## Evaluation Plan

### Phase 6A: Baseline Condition

**Task**: Evaluate all 50 problems with baseline CoT prompting

**Command:**
```bash
python scripts/run_evaluation.py \
    --condition baseline \
    --limit 50 \
    --output-dir results/phase6
```

**Expected output:**
- Results saved to: `results/phase6/baseline_results_<timestamp>.json`
- Each result contains:
  - Problem ID and category
  - Prompt used
  - Model response
  - Reasoning steps extracted
  - ME, CE, and Accuracy scores
  - Latency (ms)

**Estimated time**: ~5-10 minutes (6-12 sec/problem)

**Success criteria:**
- All 50 problems evaluated
- No crashes or errors
- Results JSON valid
- Average accuracy > 0 (sanity check)

---

### Phase 6B: MECE Condition

**Task**: Evaluate all 50 problems with MECE v1 prompting

**Command:**
```bash
python scripts/run_evaluation.py \
    --condition mece \
    --mece-version 1 \
    --limit 50 \
    --output-dir results/phase6
```

**Expected output:**
- Results saved to: `results/phase6/mece_results_<timestamp>.json`
- Same structure as baseline results

**Estimated time**: ~5-10 minutes

**Success criteria:**
- All 50 problems evaluated
- No crashes or errors
- Results JSON valid
- Average accuracy > 0 (sanity check)

---

### Phase 6C: Comparison and Validation

**Task**: Compare baseline vs MECE results

**Option 1: Run both together**
```bash
python scripts/run_evaluation.py \
    --both \
    --limit 50 \
    --output-dir results/phase6
```
This automatically shows comparison at the end.

**Manual comparison:**
After running both separately, examine:
1. Average accuracy (F1 score)
2. Average ME score
3. Average CE score
4. Average latency
5. Per-category performance

**Expected findings:**
- MECE *should* have higher ME scores (more distinct steps)
- MECE *should* have higher CE scores (better coverage)
- Accuracy could go either way (depends on model)
- Latency should be similar

---

## Monitoring and Debugging

### During Evaluation

**Monitor for:**
- Model generation errors
- Metric computation failures
- Excessive latency (>30 sec/problem)
- Memory issues

**If errors occur:**
1. Check error messages in console
2. Examine last result in JSON file
3. Re-run single problem with `--verbose` flag
4. File bug report with reproduction steps

### After Evaluation

**Validation steps:**
1. Count results: Should be exactly 50 per condition
2. Check for missing metrics in results
3. Verify latency is reasonable (<30 sec/problem)
4. Spot-check 5 random responses manually

**Quality checks:**
```bash
# Count results
python -c "import json; print(len(json.load(open('results/phase6/baseline_results_*.json'))))"

# Check for errors
python -c "
import json
results = json.load(open('results/phase6/baseline_results_*.json'))
errors = [r for r in results if 'error' in str(r['metrics'])]
print(f'Results with errors: {len(errors)}')
"

# Average scores
python -c "
import json
results = json.load(open('results/phase6/baseline_results_*.json'))
avg_acc = sum(r['metrics']['accuracy']['f1_score'] for r in results) / len(results)
print(f'Average accuracy: {avg_acc:.3f}')
"
```

---

## Expected Results Structure

Each result file contains list of dictionaries:

```json
[
  {
    "problem_id": "abs_val_001",
    "category": "absolute_value",
    "condition": "baseline",
    "mece_version": null,
    "prompt": "...",
    "response": "...",
    "reasoning_steps": ["Step 1: ...", "Step 2: ..."],
    "n_steps": 4,
    "metrics": {
      "accuracy": {
        "exact_match": true,
        "f1_score": 1.0,
        ...
      },
      "mutual_exclusivity": {
        "overall_me_score": 0.85,
        ...
      },
      "collective_exhaustiveness": {
        "overall_ce_score": 0.90,
        ...
      }
    },
    "latency_ms": 8234,
    "timestamp": "2025-11-23T..."
  },
  ...
]
```

---

## Troubleshooting

### Common Issues

**1. Model download timeout**
- First run downloads ~600MB model
- May take 5-10 minutes on slow connection
- Model cached in `~/.cache/huggingface/`

**2. Sentence-transformers download**
- ME metrics download all-MiniLM-L6-v2 (~400MB)
- Also cached after first run

**3. Out of memory**
- 0.6B model is small, should work on M4 Max
- If OOM, try reducing batch size or max_tokens

**4. Slow inference**
- Expected: 6-12 sec/problem on M4 Max
- If >30 sec, check system load
- MLX should use GPU acceleration automatically

**5. Metric computation errors**
- Check if dependencies fully installed
- Verify sentence-transformers loaded correctly
- Re-run `uv sync` if needed

---

## Post-Evaluation Checklist

After successful evaluation:

- [ ] Baseline results file exists and is valid JSON
- [ ] MECE results file exists and is valid JSON
- [ ] Both files have exactly 50 results each
- [ ] No errors in metrics
- [ ] Average accuracy > 0.1 (sanity check)
- [ ] Results backed up (optional)
- [ ] Ready for Phase 7 (Analysis)

---

## Next Steps

After Phase 6 completion:

**Phase 7: Analysis & Visualization**
- Load results into pandas DataFrames
- Compute aggregate statistics
- Statistical significance tests
- Create visualization plots
- Per-category analysis

**Phase 8: Documentation**
- Write up findings in RESULTS.md
- Document interesting cases
- Summarize insights
- Create final report

---

## Notes

- Keep both result files for comparison
- Results are gitignored (don't commit large JSON files)
- Can re-run evaluation at any time
- Consider running with different MECE versions (v1, v2, v3) for ablation study

---

## Timeline Estimate

| Task | Time | Cumulative |
|------|------|------------|
| Setup dependencies | 5-10 min | 10 min |
| Test pipeline (3 problems) | 2-3 min | 13 min |
| Baseline evaluation (50) | 8-10 min | 23 min |
| MECE evaluation (50) | 8-10 min | 33 min |
| Validation & checks | 2 min | 35 min |
| **Total** | **~35 min** | **35 min** |

*Actual time may vary based on network speed and system performance*
