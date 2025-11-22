#!/usr/bin/env python3
"""
Setup validation script for UNKNOWN Token Sink project.

This script checks that all requirements are met before beginning training.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.9+"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def check_imports():
    """Check all required packages can be imported"""
    print("\n✓ Checking package imports...")
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("tensorboard", "TensorBoard"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
    ]

    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (not installed)")
            all_ok = False

    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\n✓ Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ CUDA available")
            print(f"  ✅ {device_count} GPU(s) detected")
            print(f"  ✅ Device 0: {device_name}")
            print(f"  ✅ Memory: {memory:.1f} GB")

            if memory < 16:
                print(f"  ⚠️  Warning: <16GB VRAM may cause OOM errors")
                print(f"     Consider reducing batch size in configs/training_config.yaml")

            return True
        else:
            print("  ⚠️  CUDA not available (CPU-only mode)")
            print("     Training will be very slow without GPU")
            return True  # Not a hard failure
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False

def check_directory_structure():
    """Check all required directories exist"""
    print("\n✓ Checking directory structure...")
    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/baselines",
        "src/utils",
        "configs",
        "scripts",
        "notebooks",
        "data/train",
        "data/validation",
        "data/test",
        "output",
        "logs",
        "results",
    ]

    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (missing)")
            all_ok = False

    return all_ok

def check_config_files():
    """Check configuration files exist"""
    print("\n✓ Checking configuration files...")
    required_configs = [
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "configs/eval_config.yaml",
    ]

    all_ok = True
    for config in required_configs:
        if Path(config).exists():
            print(f"  ✅ {config}")
        else:
            print(f"  ❌ {config} (missing)")
            all_ok = False

    return all_ok

def check_documentation():
    """Check documentation files exist"""
    print("\n✓ Checking documentation...")
    docs = [
        "README.md",
        "CLAUDE.md",
        "RESEARCH.md",
        "MODEL_SELECTION.md",
        "GIBBERISH_GENERATION.md",
        "IMPLEMENTATION_DESIGN.md",
        "IMPLEMENTATION.md",
    ]

    all_ok = True
    for doc in docs:
        if Path(doc).exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ⚠️  {doc} (missing, but not critical)")

    return True  # Documentation missing is not a hard failure

def check_huggingface_cache():
    """Check Hugging Face cache directory"""
    print("\n✓ Checking Hugging Face setup...")
    try:
        from transformers import AutoTokenizer
        print("  ✅ Transformers library functional")

        # Check cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if Path(cache_dir).exists():
            print(f"  ✅ HF cache dir exists: {cache_dir}")
        else:
            print(f"  ℹ️  HF cache will be created at: {cache_dir}")

        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def print_summary(checks):
    """Print summary of all checks"""
    print("\n" + "=" * 60)
    print("SETUP VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(checks.values())
    total = len(checks)

    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")

    print("=" * 60)
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Ready to proceed with data generation.")
        print("\nNext steps:")
        print("  1. Generate data: bash scripts/generate_all_data.sh")
        print("  2. Train model: python src/training/train.py")
        print("  3. See README.md for complete workflow")
        return True
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nFor installation help, see:")
        print("  - README.md (Installation section)")
        print("  - IMPLEMENTATION.md (Phase 0: Project Setup)")
        return False

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("UNKNOWN TOKEN SINK - SETUP VALIDATION")
    print("=" * 60)
    print()

    checks = {
        "Python version": check_python_version(),
        "Required packages": check_imports(),
        "CUDA/GPU": check_cuda(),
        "Directory structure": check_directory_structure(),
        "Configuration files": check_config_files(),
        "Documentation": check_documentation(),
        "Hugging Face setup": check_huggingface_cache(),
    }

    success = print_summary(checks)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
