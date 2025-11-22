#!/usr/bin/env python3
"""
Integration test for UNKNOWN Token Sink.

Tests that all major components can be imported and initialized.
Does NOT require GPU or full data - just tests the code structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        from data.generate_gibberish import GibberishGenerator
        print("  ✓ data.generate_gibberish")
    except Exception as e:
        print(f"  ✗ data.generate_gibberish: {e}")
        return False

    try:
        from data.download_fineweb import download_fineweb_edu
        print("  ✓ data.download_fineweb")
    except Exception as e:
        print(f"  ✗ data.download_fineweb: {e}")
        return False

    try:
        from data.prepare_training_data import prepare_training_data
        print("  ✓ data.prepare_training_data")
    except Exception as e:
        print(f"  ✗ data.prepare_training_data: {e}")
        return False

    try:
        from data.dataset import MixedTrainingDataset, EvaluationDataset
        print("  ✓ data.dataset")
    except Exception as e:
        print(f"  ✗ data.dataset: {e}")
        return False

    try:
        from evaluation.metrics import compute_unknown_rate, MetricsTracker
        print("  ✓ evaluation.metrics")
    except Exception as e:
        print(f"  ✗ evaluation.metrics: {e}")
        return False

    try:
        from evaluation.evaluate import evaluate_dataset
        print("  ✓ evaluation.evaluate")
    except Exception as e:
        print(f"  ✗ evaluation.evaluate: {e}")
        return False

    # Note: Skip model and training imports as they require heavy dependencies
    # and may fail on systems without proper setup

    return True


def test_gibberish_generation():
    """Test gibberish generation."""
    print("\nTesting gibberish generation...")

    try:
        from data.generate_gibberish import GibberishGenerator

        generator = GibberishGenerator()
        examples = generator.generate_all(
            num_examples=20,
            source_texts_for_corruption=["Test sentence for corruption."] * 10
        )

        if len(examples) != 20:
            print(f"  ✗ Expected 20 examples, got {len(examples)}")
            return False

        # Check types
        types = set(ex['type'] for ex in examples)
        expected_types = {'repetitive', 'random', 'semantic_null', 'corrupted'}

        if not expected_types.issubset(types):
            print(f"  ✗ Missing types. Got: {types}")
            return False

        print(f"  ✓ Generated {len(examples)} examples with {len(types)} types")

        # Show sample
        for gtype in expected_types:
            sample = next((ex for ex in examples if ex['type'] == gtype), None)
            if sample:
                text = sample['text'][:60] + "..." if len(sample['text']) > 60 else sample['text']
                print(f"    {gtype}: {text}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics computation...")

    try:
        from evaluation.metrics import (
            compute_unknown_rate,
            evaluate_abstention_classification,
            MetricsTracker
        )

        # Test UNKNOWN rate
        predictions = ["text", "<UNKNOWN>", "more text", "<UNKNOWN>"]
        rate = compute_unknown_rate(predictions)

        if rate != 0.5:
            print(f"  ✗ UNKNOWN rate: expected 0.5, got {rate}")
            return False

        print(f"  ✓ UNKNOWN rate: {rate}")

        # Test classification metrics
        ground_truth = [False, True, False, True]
        metrics = evaluate_abstention_classification(predictions, ground_truth)

        print(f"  ✓ Classification metrics: {metrics['accuracy']:.2f} accuracy")

        # Test tracker
        tracker = MetricsTracker()
        tracker.add_metrics("test", {"rate": rate, "metrics": metrics})

        print("  ✓ Metrics tracker working")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """Test that config files exist and are valid."""
    print("\nTesting configuration files...")

    import yaml

    configs = [
        "configs/model_config.yaml",
        "configs/training_config.yaml",
        "configs/eval_config.yaml"
    ]

    project_root = Path(__file__).parent.parent

    for config_path in configs:
        full_path = project_root / config_path

        if not full_path.exists():
            print(f"  ✗ {config_path}: NOT FOUND")
            return False

        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                print(f"  ✗ {config_path}: EMPTY")
                return False

            print(f"  ✓ {config_path}")

        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
            return False

    return True


def test_scripts():
    """Test that scripts exist and are executable."""
    print("\nTesting scripts...")

    scripts = [
        "scripts/setup_environment.sh",
        "scripts/validate_setup.py",
        "scripts/generate_all_data.sh",
        "scripts/run_evaluation.sh"
    ]

    project_root = Path(__file__).parent.parent

    for script_path in scripts:
        full_path = project_root / script_path

        if not full_path.exists():
            print(f"  ✗ {script_path}: NOT FOUND")
            return False

        # Check executable for .sh files
        if script_path.endswith('.sh'):
            import os
            if not os.access(full_path, os.X_OK):
                print(f"  ✗ {script_path}: NOT EXECUTABLE")
                return False

        print(f"  ✓ {script_path}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("UNKNOWN TOKEN SINK - INTEGRATION TEST")
    print("=" * 60)
    print()

    tests = [
        ("Module imports", test_imports),
        ("Gibberish generation", test_gibberish_generation),
        ("Metrics computation", test_metrics),
        ("Configuration files", test_config_files),
        ("Scripts", test_scripts)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")

    print()

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print()
        print("✓ All integration tests passed!")
        print()
        print("Next steps:")
        print("  1. Generate data: bash scripts/generate_all_data.sh")
        print("  2. Train model: python src/training/train.py")
        print("  3. Evaluate: bash scripts/run_evaluation.sh")
        return 0
    else:
        print()
        print("✗ Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
