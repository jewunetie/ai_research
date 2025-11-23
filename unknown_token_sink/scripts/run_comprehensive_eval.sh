#!/bin/bash
# Run comprehensive evaluation including OOD datasets

set -e  # Exit on error

echo "============================================================"
echo "UNKNOWN TOKEN SINK - COMPREHENSIVE EVALUATION"
echo "============================================================"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found"
    echo "Please run: bash scripts/setup_environment.sh"
    exit 1
fi
echo ""

# Configuration
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/output/final_model}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"

echo "Configuration:"
echo "  Model directory: ${MODEL_DIR}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

# Check model exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "✗ Error: Model not found at ${MODEL_DIR}"
    echo ""
    echo "Please train a model first:"
    echo "  python src/training/train.py"
    exit 1
fi

echo "✓ Model found"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# 1. In-Distribution Evaluation
echo "============================================================"
echo "1. IN-DISTRIBUTION EVALUATION (FineWeb-Edu Test)"
echo "============================================================"
echo ""

python src/evaluation/evaluate.py \
    --model_dir "${MODEL_DIR}" \
    --config configs/eval_config.yaml \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "✓ In-distribution evaluation complete"
echo ""

# 2. SQuAD 2.0 Evaluation
echo "============================================================"
echo "2. SQuAD 2.0 EVALUATION (Answerable vs Unanswerable)"
echo "============================================================"
echo ""

python src/evaluation/evaluate_squad2.py \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_examples 1000 \
    --batch_size 8

echo ""
echo "✓ SQuAD 2.0 evaluation complete"
echo ""

# 3. TruthfulQA Evaluation
echo "============================================================"
echo "3. TruthfulQA EVALUATION (Hallucination)"
echo "============================================================"
echo ""

python src/evaluation/evaluate_truthfulqa.py \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_examples 500 \
    --batch_size 8 \
    --baseline_hallucination 0.30

echo ""
echo "✓ TruthfulQA evaluation complete"
echo ""

# 4. PubMedQA Evaluation
echo "============================================================"
echo "4. PubMedQA EVALUATION (Domain Shift)"
echo "============================================================"
echo ""

python src/evaluation/evaluate_pubmedqa.py \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_examples 500 \
    --batch_size 8

echo ""
echo "✓ PubMedQA evaluation complete"
echo ""

# Aggregate results
echo "============================================================"
echo "AGGREGATING RESULTS"
echo "============================================================"
echo ""

python -c "
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')

# Load all results
results = {}

result_files = {
    'in_distribution': 'evaluation_results.json',
    'squad2': 'squad2_results.json',
    'truthfulqa': 'truthfulqa_results.json',
    'pubmedqa': 'pubmedqa_results.json'
}

for key, filename in result_files.items():
    filepath = output_dir / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            results[key] = json.load(f)
        print(f'✓ Loaded {filename}')
    else:
        print(f'⚠️  {filename} not found')

# Save aggregated results
aggregated_file = output_dir / 'aggregated_results.json'
with open(aggregated_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Aggregated results saved to {aggregated_file}')

# Print summary
print('\n' + '='*60)
print('EVALUATION SUMMARY')
print('='*60)

if 'in_distribution' in results:
    print('\nIn-Distribution:')
    in_dist = results['in_distribution'].get('in_distribution', {})
    print(f'  UNKNOWN rate: {in_dist.get(\"unknown_rate\", 0):.2%}')
    print(f'  Target: <5%')
    print(f'  Pass: {in_dist.get(\"unknown_rate\", 1) < 0.05}')

if 'in_distribution' in results and 'synthetic_gibberish' in results['in_distribution']:
    print('\nSynthetic Gibberish:')
    gib = results['in_distribution']['synthetic_gibberish']
    print(f'  UNKNOWN rate: {gib.get(\"unknown_rate\", 0):.2%}')
    print(f'  Target: >90%')
    print(f'  Pass: {gib.get(\"unknown_rate\", 0) > 0.90}')

if 'squad2' in results:
    print('\nSQuAD 2.0:')
    squad = results['squad2']
    print(f'  Answerable UNKNOWN rate: {squad.get(\"answerable\", {}).get(\"unknown_rate\", 0):.2%} (target: <10%)')
    print(f'  Unanswerable UNKNOWN rate: {squad.get(\"unanswerable\", {}).get(\"unknown_rate\", 0):.2%} (target: >70%)')
    print(f'  F1 Score: {squad.get(\"overall\", {}).get(\"f1\", 0):.4f}')

if 'truthfulqa' in results:
    print('\nTruthfulQA:')
    tqa = results['truthfulqa']
    print(f'  Hallucination rate: {tqa.get(\"hallucination\", {}).get(\"rate\", 0):.2%}')
    print(f'  Abstention rate: {tqa.get(\"abstention\", {}).get(\"rate\", 0):.2%}')
    if 'comparison' in tqa:
        comp = tqa['comparison']
        print(f'  Hallucination reduction: {comp.get(\"relative_reduction\", 0):.2%}')
        print(f'  Pass (>5% reduction): {comp.get(\"pass\", False)}')

if 'pubmedqa' in results:
    print('\nPubMedQA (Domain Shift):')
    pqa = results['pubmedqa']
    print(f'  UNKNOWN rate: {pqa.get(\"unknown_rate\", 0):.2%}')
    print(f'  Expected range: 10-40%')

print('\n' + '='*60)
"

echo ""
echo "============================================================"
echo "COMPREHENSIVE EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "View individual results:"
echo "  cat ${OUTPUT_DIR}/evaluation_results.json | python -m json.tool"
echo "  cat ${OUTPUT_DIR}/squad2_results.json | python -m json.tool"
echo "  cat ${OUTPUT_DIR}/truthfulqa_results.json | python -m json.tool"
echo "  cat ${OUTPUT_DIR}/pubmedqa_results.json | python -m json.tool"
echo ""
echo "View aggregated results:"
echo "  cat ${OUTPUT_DIR}/aggregated_results.json | python -m json.tool"
echo ""
