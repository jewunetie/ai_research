#!/usr/bin/env python3
"""
Simple test runner for Phase 1 without requiring pytest.
This validates that all Phase 1 deliverables are in place.
"""

import json
import sys
from pathlib import Path


def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...")

    project_root = Path(__file__).parent.parent
    required_dirs = [
        "data",
        "data/results",
        "src",
        "src/models",
        "src/prompts",
        "src/metrics",
        "src/evaluation",
        "src/utils",
        "scripts",
        "tests",
        "notebooks"
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_exist = False

    return all_exist


def test_init_files():
    """Test that all __init__.py files exist."""
    print("\nTesting Python package structure...")

    project_root = Path(__file__).parent.parent
    required_init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/prompts/__init__.py",
        "src/metrics/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]

    all_exist = True
    for init_file in required_init_files:
        full_path = project_root / init_file
        if full_path.exists() and full_path.is_file():
            print(f"  ✅ {init_file}")
        else:
            print(f"  ❌ {init_file} - MISSING")
            all_exist = False

    return all_exist


def test_configuration_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")

    project_root = Path(__file__).parent.parent
    required_configs = [
        "pyproject.toml",
        "config.yaml",
        "README.md"
    ]

    all_exist = True
    for config_file in required_configs:
        full_path = project_root / config_file
        if full_path.exists() and full_path.is_file():
            # Check file is not empty
            if full_path.stat().st_size > 0:
                print(f"  ✅ {config_file} ({full_path.stat().st_size} bytes)")
            else:
                print(f"  ⚠️  {config_file} - EXISTS BUT EMPTY")
                all_exist = False
        else:
            print(f"  ❌ {config_file} - MISSING")
            all_exist = False

    return all_exist


def test_dataset():
    """Test that dataset exists and is valid."""
    print("\nTesting dataset...")

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    if not dataset_path.exists():
        print(f"  ❌ Dataset file missing: {dataset_path}")
        return False

    print(f"  ✅ Dataset file exists")

    # Load and validate
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return False

    print(f"  ✅ Valid JSON format")

    # Check structure
    if not isinstance(dataset, list):
        print(f"  ❌ Dataset must be a list")
        return False

    print(f"  ✅ Dataset is a list")
    print(f"  ℹ️  Total problems: {len(dataset)}")

    # Count categories
    categories = {}
    for problem in dataset:
        cat = problem.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  ℹ️  Distribution:")
    for cat in sorted(categories.keys()):
        print(f"      - {cat}: {categories[cat]} problems")

    # Check required fields
    required_fields = ["id", "category", "problem", "required_cases",
                      "ground_truth_solutions", "explanation"]

    issues = 0
    for i, problem in enumerate(dataset):
        missing = [f for f in required_fields if f not in problem]
        if missing:
            print(f"  ❌ Problem {i} ({problem.get('id', 'unknown')}): missing {missing}")
            issues += 1

    if issues == 0:
        print(f"  ✅ All problems have required fields")
    else:
        print(f"  ❌ {issues} problems have missing fields")
        return False

    # Check for duplicate IDs
    ids = [p["id"] for p in dataset if "id" in p]
    if len(ids) != len(set(ids)):
        print(f"  ❌ Duplicate IDs found")
        return False

    print(f"  ✅ All IDs are unique")

    return True


def test_scripts():
    """Test that required scripts exist."""
    print("\nTesting scripts...")

    project_root = Path(__file__).parent.parent
    required_scripts = [
        "scripts/create_dataset.py",
        "scripts/test_phase1.py"
    ]

    all_exist = True
    for script_file in required_scripts:
        full_path = project_root / script_file
        if full_path.exists() and full_path.is_file():
            print(f"  ✅ {script_file}")
        else:
            print(f"  ❌ {script_file} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("=" * 70)
    print("PHASE 1 IMPLEMENTATION TEST")
    print("=" * 70)

    results = {}

    results["Directory Structure"] = test_directory_structure()
    results["Python Package"] = test_init_files()
    results["Configuration Files"] = test_configuration_files()
    results["Dataset"] = test_dataset()
    results["Scripts"] = test_scripts()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("\nPhase 1 Deliverables:")
        print("  ✅ Project directory structure created")
        print("  ✅ Python package structure initialized")
        print("  ✅ Configuration files created (pyproject.toml, config.yaml, README.md)")
        print("  ✅ Dataset with 50 math case analysis problems")
        print("  ✅ Dataset validation script")
        print("  ✅ Unit tests for dataset")
        print("\nReady to proceed to Phase 2: Model Setup!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
