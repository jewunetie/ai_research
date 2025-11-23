#!/bin/bash
#
# Setup script for installing dependencies for MECE evaluation.
#
# This script:
# 1. Checks if uv is installed
# 2. Installs all dependencies via uv sync
# 3. Verifies installation succeeded
#
# Usage:
#   ./scripts/setup_dependencies.sh
#

set -e  # Exit on error

echo "======================================================================"
echo "MECE Step-by-Step: Dependency Setup"
echo "======================================================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ uv is installed: $(uv --version)"
echo ""

# Navigate to project directory
cd "$(dirname "$0")/.."
echo "Project directory: $(pwd)"
echo ""

# Run uv sync
echo "Installing dependencies with 'uv sync'..."
echo "This may take several minutes..."
echo ""

uv sync

echo ""
echo "======================================================================"
echo "✅ Dependencies installed successfully!"
echo "======================================================================"
echo ""
echo "Installed packages include:"
echo "  - MLX (Apple Silicon ML framework)"
echo "  - mlx-lm (MLX language models)"
echo "  - transformers (HuggingFace)"
echo "  - sentence-transformers (for ME metrics)"
echo "  - numpy, torch, and other ML dependencies"
echo ""
echo "Next steps:"
echo "  1. Test the full pipeline:"
echo "     python scripts/test_full_pipeline.py"
echo ""
echo "  2. If tests pass, run full evaluation:"
echo "     python scripts/run_evaluation.py --both --limit 50"
echo ""
