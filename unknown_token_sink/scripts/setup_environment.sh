#!/bin/bash
# Environment setup script for UNKNOWN Token Sink project
# This script initializes the uv virtual environment and installs all dependencies

set -e  # Exit on error

echo "=================================================="
echo "UNKNOWN Token Sink - Environment Setup"
echo "=================================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✅ uv is installed: $(uv --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
uv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install project dependencies
echo "Installing project dependencies..."
uv pip install -e .
echo "✅ Dependencies installed"
echo ""

# Install dev dependencies (optional)
echo "Installing development dependencies..."
uv pip install -e ".[dev]"
echo "✅ Development dependencies installed"
echo ""

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To validate your setup, run:"
echo "  python scripts/validate_setup.py"
echo ""
