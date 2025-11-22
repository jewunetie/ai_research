#!/usr/bin/env python3
"""
Test Phase 2 implementation: Prompts and basic infrastructure.

This tests that prompts are generated correctly without requiring the model.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import baseline_prompt, mece_prompt
from src.utils import config_simple as config


def test_config_loader():
    """Test configuration loading."""
    print("Testing configuration loader...")

    try:
        cfg = config.load_config()
        print(f"  ✅ Config loaded successfully")
        print(f"  ℹ️  Model: {cfg['model']['name']}")
        print(f"  ℹ️  Thinking mode: {cfg['model']['thinking_mode']}")
        print(f"  ℹ️  Dataset: {cfg['dataset']['path']}")
        print(f"  ℹ️  Prompt types: {cfg['evaluation']['prompt_types']}")
        return True
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False


def test_baseline_prompts():
    """Test baseline prompt generation."""
    print("\nTesting baseline prompt generation...")

    # Load sample problem
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    sample_problem = dataset[0]  # abs_val_001

    try:
        # Test basic baseline
        prompt1 = baseline_prompt.create_baseline_prompt(sample_problem)
        assert "Problem:" in prompt1
        assert sample_problem["problem"] in prompt1
        assert "step by step" in prompt1.lower()
        print(f"  ✅ Basic baseline prompt generated")

        # Test baseline with thinking
        prompt2 = baseline_prompt.create_baseline_prompt_with_thinking(sample_problem)
        assert "helpful AI" in prompt2
        assert "step by step" in prompt2.lower()
        assert sample_problem["problem"] in prompt2
        print(f"  ✅ Baseline prompt with thinking mode generated")

        # Test recommended version
        prompt3 = baseline_prompt.create_prompt(sample_problem)
        assert prompt3 == prompt2  # Should be same as with_thinking
        print(f"  ✅ Recommended baseline prompt (create_prompt) works")

        return True
    except Exception as e:
        print(f"  ❌ Baseline prompt generation failed: {e}")
        return False


def test_mece_prompts():
    """Test MECE prompt generation."""
    print("\nTesting MECE prompt generation...")

    # Load sample problem
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    sample_problem = dataset[0]  # abs_val_001

    try:
        # Test Version 1
        prompt_v1 = mece_prompt.create_mece_prompt_v1(sample_problem)
        assert "MECE" in prompt_v1
        assert "mutually exclusive" in prompt_v1.lower()
        assert "collectively exhaustive" in prompt_v1.lower()
        assert sample_problem["problem"] in prompt_v1
        print(f"  ✅ MECE prompt v1 (explicit MECE) generated")

        # Test Version 2
        prompt_v2 = mece_prompt.create_mece_prompt_v2(sample_problem)
        assert "cases" in prompt_v2.lower()
        assert "Step 1:" in prompt_v2
        assert sample_problem["problem"] in prompt_v2
        print(f"  ✅ MECE prompt v2 (case analysis template) generated")

        # Test Version 3
        prompt_v3 = mece_prompt.create_mece_prompt_v3(sample_problem)
        assert "CASE IDENTIFICATION" in prompt_v3
        assert "MUTUAL EXCLUSIVITY CHECK" in prompt_v3
        assert "EXHAUSTIVENESS CHECK" in prompt_v3
        assert sample_problem["problem"] in prompt_v3
        print(f"  ✅ MECE prompt v3 (highly structured) generated")

        # Test version selector
        for version in [1, 2, 3]:
            prompt = mece_prompt.create_mece_prompt(sample_problem, version=version)
            assert sample_problem["problem"] in prompt
        print(f"  ✅ MECE prompt version selector works")

        # Test default version
        prompt_default = mece_prompt.create_prompt(sample_problem)
        assert prompt_default == prompt_v1  # Should default to v1
        print(f"  ✅ Default MECE prompt (create_prompt) works")

        return True
    except Exception as e:
        print(f"  ❌ MECE prompt generation failed: {e}")
        return False


def test_prompt_differences():
    """Test that MECE and baseline prompts are different."""
    print("\nTesting prompt differentiation...")

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    sample_problem = dataset[0]

    try:
        baseline = baseline_prompt.create_prompt(sample_problem)
        mece = mece_prompt.create_prompt(sample_problem)

        # They should be different
        assert baseline != mece
        print(f"  ✅ Baseline and MECE prompts are different")

        # MECE should be longer (more instructions)
        assert len(mece) > len(baseline)
        print(f"  ✅ MECE prompt is longer (more detailed instructions)")

        # Baseline shouldn't mention MECE
        assert "MECE" not in baseline
        assert "mutually exclusive" not in baseline.lower()
        print(f"  ✅ Baseline doesn't mention MECE concepts")

        # MECE should mention MECE
        assert ("MECE" in mece or "mutually exclusive" in mece.lower())
        print(f"  ✅ MECE prompt includes MECE guidance")

        return True
    except Exception as e:
        print(f"  ❌ Prompt differentiation test failed: {e}")
        return False


def display_prompt_examples():
    """Display example prompts for inspection."""
    print("\n" + "=" * 70)
    print("EXAMPLE PROMPTS")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    sample_problem = dataset[0]

    print("\n📝 BASELINE PROMPT:")
    print("-" * 70)
    print(baseline_prompt.create_prompt(sample_problem))

    print("\n📝 MECE PROMPT (Version 1):")
    print("-" * 70)
    print(mece_prompt.create_mece_prompt_v1(sample_problem))

    print("\n📝 MECE PROMPT (Version 2):")
    print("-" * 70)
    print(mece_prompt.create_mece_prompt_v2(sample_problem))

    print("\n" + "=" * 70)


def main():
    """Run all Phase 2 tests."""
    print("=" * 70)
    print("PHASE 2 IMPLEMENTATION TEST")
    print("=" * 70)

    results = {}

    results["Config Loader"] = test_config_loader()
    results["Baseline Prompts"] = test_baseline_prompts()
    results["MECE Prompts"] = test_mece_prompts()
    results["Prompt Differentiation"] = test_prompt_differences()

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
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("\nPhase 2 Deliverables:")
        print("  ✅ Configuration loader implemented")
        print("  ✅ Baseline prompt templates created")
        print("  ✅ MECE prompt templates created (3 versions)")
        print("  ✅ Prompt differentiation verified")
        print("\nNext steps:")
        print("  1. Install dependencies: uv sync")
        print("  2. Test model inference (requires MLX + model download)")
        print("  3. Proceed to Phase 3: Metrics implementation")

        # Display examples
        display_prompt_examples()

        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
