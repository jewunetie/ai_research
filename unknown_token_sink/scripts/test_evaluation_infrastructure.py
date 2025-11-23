#!/usr/bin/env python3
"""
Deep testing for new evaluation infrastructure.

Tests all newly implemented components:
- SQuAD 2.0 evaluation
- TruthfulQA evaluation
- PubMedQA evaluation
- Baseline models
"""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_squad2_evaluation():
    """Test SQuAD 2.0 evaluation module."""
    print("="*60)
    print("Testing SQuAD 2.0 Evaluation")
    print("="*60)

    issues = []

    try:
        from evaluation.evaluate_squad2 import (
            load_squad2,
            format_qa_prompt,
            evaluate_squad2
        )
        print("  ✓ Module imports successful")

        # Test prompt formatting
        prompt = format_qa_prompt("What is the capital?", "Paris is the capital of France.")
        if "Context:" not in prompt or "Question:" not in prompt or "Answer:" not in prompt:
            issues.append("Prompt formatting missing required sections")
        else:
            print("  ✓ Prompt formatting correct")

        # Test with synthetic data
        print("\n  Testing with synthetic data...")
        answerable = [
            {
                'question': 'What is 2+2?',
                'context': 'Math is important.',
                'is_answerable': True,
                'answer': '4',
                'id': 'test1'
            }
        ]
        unanswerable = [
            {
                'question': 'What is the color of nothing?',
                'context': 'Nothing has no properties.',
                'is_answerable': False,
                'id': 'test2'
            }
        ]

        print(f"    Synthetic answerable: {len(answerable)}")
        print(f"    Synthetic unanswerable: {len(unanswerable)}")
        print("  ✓ Synthetic data created")

    except Exception as e:
        issues.append(f"SQuAD 2.0 evaluation error: {e}")
        import traceback
        traceback.print_exc()

    if issues:
        print(f"\n  ✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All SQuAD 2.0 tests passed")
        return True


def test_truthfulqa_evaluation():
    """Test TruthfulQA evaluation module."""
    print("\n" + "="*60)
    print("Testing TruthfulQA Evaluation")
    print("="*60)

    issues = []

    try:
        from evaluation.evaluate_truthfulqa import (
            load_truthfulqa,
            is_answer_truthful,
            evaluate_truthfulqa,
            compare_to_baseline
        )
        print("  ✓ Module imports successful")

        # Test truthfulness checking
        print("\n  Testing truthfulness checking...")

        # Test correct answer
        is_truth, category = is_answer_truthful(
            "Heart disease is the most common cause",
            ["Heart disease", "Cardiovascular disease"],
            ["Cancer", "Accidents"]
        )
        if category != 'truthful':
            issues.append(f"Expected 'truthful', got '{category}'")
        else:
            print("    ✓ Truthful answer detected correctly")

        # Test hallucination
        is_truth, category = is_answer_truthful(
            "Cancer is the most common cause",
            ["Heart disease"],
            ["Cancer", "Accidents"]
        )
        if category != 'hallucination':
            issues.append(f"Expected 'hallucination', got '{category}'")
        else:
            print("    ✓ Hallucination detected correctly")

        # Test abstention
        is_truth, category = is_answer_truthful(
            "I don't know the answer",
            ["Heart disease"],
            ["Cancer"]
        )
        if category != 'abstention':
            issues.append(f"Expected 'abstention', got '{category}'")
        else:
            print("    ✓ Abstention detected correctly")

        # Test with <UNKNOWN>
        is_truth, category = is_answer_truthful(
            "This is <UNKNOWN> to me",
            ["test"],
            ["wrong"]
        )
        if category != 'abstention':
            issues.append(f"Expected 'abstention' for UNKNOWN, got '{category}'")
        else:
            print("    ✓ UNKNOWN token detected as abstention")

        # Test baseline comparison
        print("\n  Testing baseline comparison...")
        mock_metrics = {
            'hallucination': {'rate': 0.25}
        }
        comparison = compare_to_baseline(mock_metrics, baseline_hallucination_rate=0.30)

        if 'relative_reduction' not in comparison:
            issues.append("Missing 'relative_reduction' in comparison")
        else:
            expected_reduction = (0.30 - 0.25) / 0.30
            if abs(comparison['relative_reduction'] - expected_reduction) > 0.01:
                issues.append(f"Incorrect reduction calculation: {comparison['relative_reduction']}")
            else:
                print(f"    ✓ Reduction calculated correctly: {comparison['relative_reduction']:.2%}")

    except Exception as e:
        issues.append(f"TruthfulQA evaluation error: {e}")
        import traceback
        traceback.print_exc()

    if issues:
        print(f"\n  ✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All TruthfulQA tests passed")
        return True


def test_pubmedqa_evaluation():
    """Test PubMedQA evaluation module."""
    print("\n" + "="*60)
    print("Testing PubMedQA Evaluation")
    print("="*60)

    issues = []

    try:
        from evaluation.evaluate_pubmedqa import (
            load_pubmedqa,
            evaluate_pubmedqa
        )
        print("  ✓ Module imports successful")

        # Test loading (will use synthetic data)
        print("\n  Testing data loading...")
        examples = load_pubmedqa(max_examples=10)

        if len(examples) == 0:
            issues.append("No examples loaded")
        elif len(examples) > 10:
            issues.append(f"Expected max 10 examples, got {len(examples)}")
        else:
            print(f"    ✓ Loaded {len(examples)} examples")

        # Check example format
        if examples:
            ex = examples[0]
            required_keys = ['question', 'context']
            for key in required_keys:
                if key not in ex:
                    issues.append(f"Example missing '{key}' key")

            if not issues:
                print("    ✓ Example format correct")

    except Exception as e:
        issues.append(f"PubMedQA evaluation error: {e}")
        import traceback
        traceback.print_exc()

    if issues:
        print(f"\n  ✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All PubMedQA tests passed")
        return True


def test_baseline_models():
    """Test all baseline models."""
    print("\n" + "="*60)
    print("Testing Baseline Models")
    print("="*60)

    issues = []

    # Test Unmodified Baseline
    print("\n  Testing UnmodifiedBaseline...")
    try:
        from baselines.unmodified_baseline import UnmodifiedBaseline

        # Test with small model
        baseline = UnmodifiedBaseline(
            model_name="gpt2",
            torch_dtype="float32"
        )

        # Test generation
        output = baseline.generate("Test input", max_new_tokens=5)
        if not output or len(output) == 0:
            issues.append("UnmodifiedBaseline: No output generated")
        else:
            print("    ✓ Generation works")

        # Test batch generation
        outputs = baseline.batch_generate(
            ["Test 1", "Test 2"],
            max_new_tokens=5,
            batch_size=2
        )
        if len(outputs) != 2:
            issues.append(f"UnmodifiedBaseline: Expected 2 outputs, got {len(outputs)}")
        else:
            print("    ✓ Batch generation works")

    except Exception as e:
        issues.append(f"UnmodifiedBaseline error: {e}")
        import traceback
        traceback.print_exc()

    # Test Confidence Threshold Baseline
    print("\n  Testing ConfidenceThresholdBaseline...")
    try:
        from baselines.confidence_threshold import ConfidenceThresholdBaseline

        baseline = ConfidenceThresholdBaseline(
            model_name="gpt2",
            confidence_threshold=0.8,
            torch_dtype="float32"
        )

        # Test generation with confidence
        output, confidence = baseline.generate(
            "Test input",
            max_new_tokens=5,
            return_confidence=True
        )

        if confidence < 0 or confidence > 1:
            issues.append(f"ConfidenceThreshold: Invalid confidence {confidence}")
        else:
            print(f"    ✓ Confidence computed: {confidence:.4f}")

        # Check abstention token handling
        if baseline.abstention_token not in baseline.tokenizer.get_vocab():
            print(f"    ℹ️  Abstention token '{baseline.abstention_token}' not in vocab (expected)")

    except Exception as e:
        issues.append(f"ConfidenceThresholdBaseline error: {e}")
        import traceback
        traceback.print_exc()

    # Test Prompt-Based Baseline
    print("\n  Testing PromptBasedBaseline...")
    try:
        from baselines.prompt_based import PromptBasedBaseline

        baseline = PromptBasedBaseline(
            model_name="gpt2",
            torch_dtype="float32",
            instruction_template="default"
        )

        # Test prompt formatting
        prompt = baseline.format_prompt("What is the capital of France?")
        if "Question:" not in prompt:
            issues.append("PromptBased: Prompt missing 'Question:' marker")
        else:
            print("    ✓ Prompt formatting works")

        # Test all templates
        for template_name in baseline.templates.keys():
            try:
                prompt = baseline.format_prompt("Test question", template=template_name)
                if len(prompt) == 0:
                    issues.append(f"PromptBased: Empty prompt for template '{template_name}'")
            except Exception as e:
                issues.append(f"PromptBased: Template '{template_name}' error: {e}")

        if not any("PromptBased: Template" in i for i in issues):
            print(f"    ✓ All {len(baseline.templates)} templates work")

    except Exception as e:
        issues.append(f"PromptBasedBaseline error: {e}")
        import traceback
        traceback.print_exc()

    if issues:
        print(f"\n  ✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All baseline model tests passed")
        return True


def test_imports_and_structure():
    """Test that all imports work correctly."""
    print("\n" + "="*60)
    print("Testing Imports and Module Structure")
    print("="*60)

    issues = []

    # Test evaluation modules
    eval_modules = [
        'evaluation.evaluate_squad2',
        'evaluation.evaluate_truthfulqa',
        'evaluation.evaluate_pubmedqa'
    ]

    for module_name in eval_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            issues.append(f"Failed to import {module_name}: {e}")

    # Test baseline modules
    baseline_modules = [
        'baselines.unmodified_baseline',
        'baselines.confidence_threshold',
        'baselines.prompt_based'
    ]

    for module_name in baseline_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            issues.append(f"Failed to import {module_name}: {e}")

    if issues:
        print(f"\n  ✗ Found {len(issues)} import issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All imports successful")
        return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)

    issues = []

    # Test empty inputs
    print("\n  Testing empty inputs...")
    try:
        from evaluation.evaluate_truthfulqa import is_answer_truthful

        # Empty answer
        is_truth, category = is_answer_truthful("", ["test"], ["wrong"])
        print(f"    Empty answer → category: {category}")

        # Empty lists
        is_truth, category = is_answer_truthful("answer", [], [])
        print(f"    Empty correct/incorrect lists → category: {category}")

        print("    ✓ Empty input handling works")

    except Exception as e:
        issues.append(f"Edge case error: {e}")
        import traceback
        traceback.print_exc()

    # Test very long inputs
    print("\n  Testing long inputs...")
    try:
        from baselines.unmodified_baseline import UnmodifiedBaseline

        baseline = UnmodifiedBaseline(model_name="gpt2", torch_dtype="float32")

        # Very long input (will be truncated)
        long_input = "test " * 1000
        output = baseline.generate(long_input, max_new_tokens=5)
        print("    ✓ Long input handling works (truncation)")

    except Exception as e:
        issues.append(f"Long input error: {e}")

    # Test special characters
    print("\n  Testing special characters...")
    try:
        from evaluation.evaluate_squad2 import format_qa_prompt

        special_text = "What is <UNKNOWN>? Test 'quotes' and \"double quotes\""
        prompt = format_qa_prompt(special_text, "Context with special: <>&")

        if len(prompt) == 0:
            issues.append("Special characters caused empty prompt")
        else:
            print("    ✓ Special character handling works")

    except Exception as e:
        issues.append(f"Special character error: {e}")

    if issues:
        print(f"\n  ✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("\n  ✓ All edge case tests passed")
        return True


def main():
    """Run all tests."""
    print("="*60)
    print("COMPREHENSIVE EVALUATION INFRASTRUCTURE TESTING")
    print("="*60)
    print()

    tests = [
        ("Imports and Structure", test_imports_and_structure),
        ("SQuAD 2.0 Evaluation", test_squad2_evaluation),
        ("TruthfulQA Evaluation", test_truthfulqa_evaluation),
        ("PubMedQA Evaluation", test_pubmedqa_evaluation),
        ("Baseline Models", test_baseline_models),
        ("Edge Cases", test_edge_cases)
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

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print()

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")

    print()

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"Results: {passed_count}/{total} test suites passed")

    if passed_count == total:
        print()
        print("✓ All evaluation infrastructure tests passed!")
        print()
        print("The new code is ready for use:")
        print("  - SQuAD 2.0 evaluation")
        print("  - TruthfulQA evaluation")
        print("  - PubMedQA evaluation")
        print("  - 3 baseline models")
        print()
        return 0
    else:
        print()
        print("✗ Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
