#!/bin/bash
# Run all experiment configurations

echo "==========================================="
echo "Running All Experiment Configurations"
echo "==========================================="

# Array of experiment configs
experiments=(
    "config/experiments/baseline.yaml"
    "config/experiments/high_labor_preference.yaml"
    "config/experiments/high_pricing.yaml"
)

# Run each experiment
for config in "${experiments[@]}"; do
    exp_name=$(basename "$config" .yaml)
    echo ""
    echo "-------------------------------------------"
    echo "Running: $exp_name"
    echo "-------------------------------------------"

    python src/main.py --config "$config"

    if [ $? -eq 0 ]; then
        echo "✓ $exp_name completed successfully"
    else
        echo "✗ $exp_name failed"
    fi
done

echo ""
echo "==========================================="
echo "All experiments complete!"
echo "==========================================="
echo ""
echo "Results saved to ./results/"
