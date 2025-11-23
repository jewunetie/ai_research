#!/usr/bin/env python3
"""
Phase 6 Preparation: Test full pipeline with actual model before running complete evaluation.

This script:
1. Checks dependencies are installed
2. Tests model loading
3. Runs end-to-end evaluation on 3 test problems
4. Validates all metrics (ME, CE, Accuracy) work correctly
5. Reports any issues before full evaluation

Usage:
    # After running: uv sync
    python scripts/test_full_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 70)
    print("STEP 1: Checking Dependencies")
    print("=" * 70)

    missing = []

    # Check MLX
    try:
        import mlx
        import mlx_lm
        print(f"✅ MLX installed (version {mlx.__version__})")
    except ImportError:
        print("❌ MLX not installed")
        missing.append("mlx")

    # Check numpy
    try:
        import numpy as np
        print(f"✅ NumPy installed (version {np.__version__})")
    except ImportError:
        print("❌ NumPy not installed")
        missing.append("numpy")

    # Check sentence-transformers
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers installed (version {sentence_transformers.__version__})")
    except ImportError:
        print("❌ sentence-transformers not installed")
        missing.append("sentence-transformers")

    # Check torch
    try:
        import torch
        print(f"✅ PyTorch installed (version {torch.__version__})")
    except ImportError:
        print("❌ PyTorch not installed")
        missing.append("torch")

    # Check transformers
    try:
        import transformers
        print(f"✅ transformers installed (version {transformers.__version__})")
    except ImportError:
        print("❌ transformers not installed")
        missing.append("transformers")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nTo install dependencies, run:")
        print("  cd mece_step_by_step && uv sync")
        return False

    print("\n✅ All dependencies installed!")
    return True


def test_model_loading():
    """Test loading Qwen3-0.6B-Instruct model."""
    print("\n" + "=" * 70)
    print("STEP 2: Testing Model Loading")
    print("=" * 70)

    try:
        from src.models.qwen_inference import QwenInference

        print("Loading Qwen3-0.6B-Instruct model...")
        print("(This may take a few minutes on first run - downloading ~600MB)")

        model = QwenInference()

        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(model):
    """Test basic model inference."""
    print("\n" + "=" * 70)
    print("STEP 3: Testing Model Inference")
    print("=" * 70)

    test_prompt = """You are a helpful AI that solves math problems step by step.

Problem: Solve the equation |x - 3| = 5

Let's solve this step by step:"""

    print(f"Test prompt (first 100 chars): {test_prompt[:100]}...")

    try:
        print("\nGenerating response...")
        response = model.generate(test_prompt, max_tokens=256)

        print(f"\n✅ Inference successful!")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview:\n{response[:200]}...\n")

        return response

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_metrics_with_dependencies():
    """Test that ME and CE metrics work with dependencies installed."""
    print("\n" + "=" * 70)
    print("STEP 4: Testing Metrics with Dependencies")
    print("=" * 70)

    # Test response
    test_response = """Step 1: When x - 3 >= 0 (i.e., x >= 3), we have x - 3 = 5
This gives x = 8

Step 2: When x - 3 < 0 (i.e., x < 3), we have -(x - 3) = 5
This gives -x + 3 = 5, so x = -2

Therefore, the solutions are x = 8 and x = -2"""

    test_problem = {
        "required_cases": ["x >= 3", "x < 3"],
        "ground_truth_solutions": ["x = 8", "x = -2"]
    }

    all_passed = True

    # Test Accuracy (always available)
    try:
        from src.metrics import compute_accuracy
        scores = compute_accuracy(test_response, test_problem["ground_truth_solutions"])
        print(f"✅ Accuracy metric: F1={scores['f1_score']:.3f}")
    except Exception as e:
        print(f"❌ Accuracy metric failed: {e}")
        all_passed = False

    # Test ME (requires numpy + sentence-transformers)
    try:
        from src.metrics import compute_mutual_exclusivity

        if compute_mutual_exclusivity is None:
            print("❌ ME metric: Not available (dependencies not installed)")
            all_passed = False
        else:
            scores = compute_mutual_exclusivity(test_response)
            print(f"✅ ME metric: Score={scores['overall_me_score']:.3f}")
    except Exception as e:
        print(f"❌ ME metric failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test CE (requires parsers)
    try:
        from src.metrics import compute_collective_exhaustiveness

        if compute_collective_exhaustiveness is None:
            print("❌ CE metric: Not available (dependencies not installed)")
            all_passed = False
        else:
            scores = compute_collective_exhaustiveness(test_response, test_problem)
            print(f"✅ CE metric: Score={scores['overall_ce_score']:.3f}")
    except Exception as e:
        print(f"❌ CE metric failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_end_to_end_evaluation(model):
    """Test end-to-end evaluation on 3 problems (both conditions)."""
    print("\n" + "=" * 70)
    print("STEP 5: Testing End-to-End Evaluation")
    print("=" * 70)
    print("Testing both BASELINE and MECE conditions on 3 problems total\n")

    try:
        from src.evaluation import MECEEvaluator

        # Load dataset
        dataset_path = Path("data/math_case_analysis.json")
        with open(dataset_path) as f:
            problems = json.load(f)

        # Take first 3 problems
        test_problems = problems[:3]

        # Create evaluator with actual model
        evaluator = MECEEvaluator(
            model_inference=model,
            mece_version=1,
            verbose=False,  # Less verbose for cleaner output
            output_dir=Path("results/test")
        )

        all_results = {}

        # Test BASELINE condition (2 problems)
        print("Testing BASELINE condition (2 problems)...")

        baseline_results = []
        for i, problem in enumerate(test_problems[:2], 1):
            print(f"  Problem {i}/2: {problem['id']}", end=" ")

            result = evaluator.evaluate_problem(problem, condition="baseline")
            baseline_results.append(result)

            # Show inline results
            acc = result['metrics']['accuracy']['f1_score']
            print(f"→ Steps: {result['n_steps']}, Acc: {acc:.2f}", end="")

            if 'mutual_exclusivity' in result['metrics'] and 'error' not in result['metrics']['mutual_exclusivity']:
                me = result['metrics']['mutual_exclusivity']['overall_me_score']
                print(f", ME: {me:.2f}", end="")

            if 'collective_exhaustiveness' in result['metrics'] and 'error' not in result['metrics']['collective_exhaustiveness']:
                ce = result['metrics']['collective_exhaustiveness']['overall_ce_score']
                print(f", CE: {ce:.2f}", end="")

            print(" ✅")

        all_results['baseline'] = baseline_results

        # Test MECE condition (1 problem)
        print("\nTesting MECE condition (1 problem)...")

        mece_results = []
        problem = test_problems[2]  # Third problem
        print(f"  Problem 1/1: {problem['id']}", end=" ")

        result = evaluator.evaluate_problem(problem, condition="mece")
        mece_results.append(result)

        # Show inline results
        acc = result['metrics']['accuracy']['f1_score']
        print(f"→ Steps: {result['n_steps']}, Acc: {acc:.2f}", end="")

        if 'mutual_exclusivity' in result['metrics'] and 'error' not in result['metrics']['mutual_exclusivity']:
            me = result['metrics']['mutual_exclusivity']['overall_me_score']
            print(f", ME: {me:.2f}", end="")

        if 'collective_exhaustiveness' in result['metrics'] and 'error' not in result['metrics']['collective_exhaustiveness']:
            ce = result['metrics']['collective_exhaustiveness']['overall_ce_score']
            print(f", CE: {ce:.2f}", end="")

        print(" ✅")

        all_results['mece'] = mece_results

        # Summary
        print(f"\n{'='*70}")
        print("TEST EVALUATION SUMMARY")
        print(f"{'='*70}")

        baseline_avg_acc = sum(r['metrics']['accuracy']['f1_score'] for r in baseline_results) / len(baseline_results)
        mece_avg_acc = sum(r['metrics']['accuracy']['f1_score'] for r in mece_results) / len(mece_results)

        print(f"Baseline (2 problems): Avg Accuracy = {baseline_avg_acc:.3f}")

        baseline_avg_latency = sum(r['latency_ms'] for r in baseline_results) / len(baseline_results)
        print(f"                        Avg Latency = {baseline_avg_latency:.0f}ms")

        print(f"MECE (1 problem):      Avg Accuracy = {mece_avg_acc:.3f}")

        mece_avg_latency = sum(r['latency_ms'] for r in mece_results) / len(mece_results)
        print(f"                        Avg Latency = {mece_avg_latency:.0f}ms")

        print(f"\n✅ End-to-end evaluation successful for BOTH conditions!")
        return True

    except Exception as e:
        print(f"❌ End-to-end evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all preparation tests."""
    print("\n" + "=" * 70)
    print("PHASE 6 PREPARATION: Full Pipeline Testing")
    print("=" * 70)
    print("\nThis script tests the complete evaluation pipeline with actual model")
    print("before running the full 50-problem evaluation.\n")

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n" + "=" * 70)
        print("❌ PREPARATION FAILED: Missing dependencies")
        print("=" * 70)
        return 1

    # Step 2: Load model
    model = test_model_loading()
    if model is None:
        print("\n" + "=" * 70)
        print("❌ PREPARATION FAILED: Model loading failed")
        print("=" * 70)
        return 1

    # Step 3: Test inference
    response = test_inference(model)
    if response is None:
        print("\n" + "=" * 70)
        print("❌ PREPARATION FAILED: Inference failed")
        print("=" * 70)
        return 1

    # Step 4: Test metrics
    if not test_metrics_with_dependencies():
        print("\n" + "=" * 70)
        print("❌ PREPARATION FAILED: Metrics testing failed")
        print("=" * 70)
        return 1

    # Step 5: Test end-to-end
    if not test_end_to_end_evaluation(model):
        print("\n" + "=" * 70)
        print("❌ PREPARATION FAILED: End-to-end evaluation failed")
        print("=" * 70)
        return 1

    # Success!
    print("\n" + "=" * 70)
    print("✅ ALL PREPARATION TESTS PASSED!")
    print("=" * 70)
    print("\nThe full pipeline is working correctly.")
    print("\nNext steps:")
    print("  1. Review the test results above")
    print("  2. Run full evaluation on 50 problems:")
    print("     python scripts/run_evaluation.py --both --limit 50")
    print("  3. Analyze results in Phase 7")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
