#!/bin/bash
# Generate all data for UNKNOWN Token Sink training
# This script orchestrates the complete data generation pipeline

set -e  # Exit on error

echo "============================================================"
echo "UNKNOWN TOKEN SINK - DATA GENERATION PIPELINE"
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
DATA_DIR="${PROJECT_ROOT}/data"
CONFIG_FILE="${PROJECT_ROOT}/configs/training_config.yaml"

echo "Configuration:"
echo "  Data directory: ${DATA_DIR}"
echo "  Config file: ${CONFIG_FILE}"
echo ""

# Step 1: Download FineWeb-Edu
echo "============================================================"
echo "STEP 1: Download FineWeb-Edu Dataset"
echo "============================================================"
echo ""

python src/data/download_fineweb.py \
    --output_dir "${DATA_DIR}" \
    --num_train 90000 \
    --num_validation 10000 \
    --num_test 5000 \
    --seed 42

echo ""
echo "✓ FineWeb-Edu download complete"
echo ""

# Step 2: Prepare training data (mix with gibberish)
echo "============================================================"
echo "STEP 2: Prepare Training Data (Mix with Gibberish)"
echo "============================================================"
echo ""

python src/data/prepare_training_data.py \
    --data_dir "${DATA_DIR}" \
    --config "${CONFIG_FILE}" \
    --seed 42

echo ""
echo "✓ Training data preparation complete"
echo ""

# Step 3: Verify data
echo "============================================================"
echo "STEP 3: Verify Generated Data"
echo "============================================================"
echo ""

# Check files exist
files_to_check=(
    "fineweb_edu_train.jsonl"
    "fineweb_edu_validation.jsonl"
    "fineweb_edu_test.jsonl"
    "train/mixed_training_data.jsonl"
    "validation/validation_data.jsonl"
    "test/test_real.jsonl"
    "test/gibberish_synthetic.jsonl"
)

all_exist=true
for file in "${files_to_check[@]}"; do
    filepath="${DATA_DIR}/${file}"
    if [ -f "$filepath" ]; then
        lines=$(wc -l < "$filepath")
        size=$(du -h "$filepath" | cut -f1)
        echo "✓ ${file}: ${lines} lines, ${size}"
    else
        echo "✗ ${file}: NOT FOUND"
        all_exist=false
    fi
done

echo ""

if [ "$all_exist" = true ]; then
    echo "============================================================"
    echo "DATA GENERATION COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Train model: python src/training/train.py"
    echo "  2. See README.md for complete workflow"
    echo ""
else
    echo "⚠️  Some files are missing. Please check errors above."
    exit 1
fi
