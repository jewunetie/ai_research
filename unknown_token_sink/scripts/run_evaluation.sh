#!/bin/bash
# Run comprehensive evaluation of trained UNKNOWN Token Sink model

set -e  # Exit on error

echo "============================================================"
echo "UNKNOWN TOKEN SINK - MODEL EVALUATION"
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
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/eval_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"

echo "Configuration:"
echo "  Model directory: ${MODEL_DIR}"
echo "  Data directory: ${DATA_DIR}"
echo "  Config file: ${CONFIG_FILE}"
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

# Check data exists
required_files=(
    "test/test_real.jsonl"
    "test/gibberish_synthetic.jsonl"
)

all_exist=true
for file in "${required_files[@]}"; do
    filepath="${DATA_DIR}/${file}"
    if [ ! -f "$filepath" ]; then
        echo "✗ Required data file not found: ${file}"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "Please generate test data first:"
    echo "  bash scripts/generate_all_data.sh"
    exit 1
fi

echo "✓ Test data found"
echo ""

# Run evaluation
echo "============================================================"
echo "RUNNING EVALUATION"
echo "============================================================"
echo ""

python src/evaluation/evaluate.py \
    --model_dir "${MODEL_DIR}" \
    --config "${CONFIG_FILE}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To view detailed results:"
echo "  cat ${OUTPUT_DIR}/evaluation_results.json | python -m json.tool"
echo ""
