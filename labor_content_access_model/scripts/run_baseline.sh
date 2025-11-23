#!/bin/bash
# Run baseline simulation

echo "========================================"
echo "Running Baseline Simulation"
echo "========================================"

# Run with baseline config
python src/main.py --config config/experiments/baseline.yaml

echo ""
echo "Baseline simulation complete!"
echo "Results saved to ./results/baseline/"
