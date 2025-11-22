#!/usr/bin/env python3
"""
Unit tests for Phase 3 metrics implementation.

Tests parsers, mutual exclusivity, collective exhaustiveness, and accuracy metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    parse_reasoning_steps,
    extract_case_conditions,
    count_overlapping_conditions,
    parse_solutions,
    normalize_solution,
    compute_mutual_exclusivity,
    compute_collective_exhaustiveness,
    compute_accuracy,
)


def test_parse_reasoning_steps():
    """Test reasoning step parsing."""
    print("Testing parse_reasoning_steps...")

    # Test numbered steps
    response1 = """
    1. First, check x >= 0
    2. Then, check x < 0
    3. Combine results
    """
    steps1 = parse_reasoning_steps(response1)
    assert len(steps1) == 3, f"Expected 3 steps, got {len(steps1)}"
    assert "check x >= 0" in steps1[0].lower()
    print("  ✅ Numbered steps parsed correctly")

    # Test "Step N:" format
    response2 = """
    Step 1: When x >= 0, solve x = 2x - 3
    Step 2: When x < 0, solve -x = 2x - 3
    """
    steps2 = parse_reasoning_steps(response2)
    assert len(steps2) == 2, f"Expected 2 steps, got {len(steps2)}"
    print("  ✅ 'Step N:' format parsed correctly")

    # Test bullet points
    response3 = """
    - Case 1: x >= 0 gives x = 3
    - Case 2: x < 0 gives x = -1
    """
    steps3 = parse_reasoning_steps(response3)
    assert len(steps3) == 2, f"Expected 2 steps, got {len(steps3)}"
    print("  ✅ Bullet points parsed correctly")

    # Test empty response
    steps4 = parse_reasoning_steps("")
    assert len(steps4) == 0, f"Expected 0 steps for empty response, got {len(steps4)}"
    print("  ✅ Empty response handled correctly")

    return True


def test_extract_case_conditions():
    """Test case condition extraction."""
    print("\nTesting extract_case_conditions...")

    steps1 = [
        "When x >= 0, we have x = 2x - 3",
        "When x < 0, we have -x = 2x - 3"
    ]
    conditions1 = extract_case_conditions(steps1)
    assert len(conditions1) >= 2, f"Expected at least 2 conditions, got {len(conditions1)}"
    assert any("x >=" in c or "x >=" in c for c in conditions1)
    assert any("x <" in c for c in conditions1)
    print("  ✅ Basic conditions extracted correctly")

    # Test different formats
    steps2 = [
        "For x > 5, calculate f(x) = x + 1",
        "If x <= 5, calculate f(x) = 2x"
    ]
    conditions2 = extract_case_conditions(steps2)
    assert len(conditions2) >= 2, f"Expected at least 2 conditions, got {len(conditions2)}"
    print("  ✅ Different condition formats extracted")

    return True


def test_count_overlapping_conditions():
    """Test overlap detection in conditions."""
    print("\nTesting count_overlapping_conditions...")

    # Overlapping conditions
    conditions1 = ["x >= 0", "x > 0"]  # These overlap
    overlaps1 = count_overlapping_conditions(conditions1)
    assert overlaps1 > 0, f"Expected overlaps, got {overlaps1}"
    print("  ✅ Overlapping conditions detected")

    # Non-overlapping conditions
    conditions2 = ["x >= 0", "x < 0"]  # These don't overlap
    overlaps2 = count_overlapping_conditions(conditions2)
    assert overlaps2 == 0, f"Expected no overlaps, got {overlaps2}"
    print("  ✅ Non-overlapping conditions recognized")

    # Edge case: single condition
    conditions3 = ["x >= 0"]
    overlaps3 = count_overlapping_conditions(conditions3)
    assert overlaps3 == 0, f"Expected no overlaps for single condition, got {overlaps3}"
    print("  ✅ Single condition handled correctly")

    return True


def test_parse_solutions():
    """Test solution parsing."""
    print("\nTesting parse_solutions...")

    # Test basic solutions
    text1 = "The solution is x = 3"
    solutions1 = parse_solutions(text1)
    assert len(solutions1) == 1, f"Expected 1 solution, got {len(solutions1)}"
    assert "x = 3" in str(solutions1)
    print("  ✅ Basic solution parsed")

    # Test multiple solutions
    text2 = "The solutions are x = 3 and x = -1"
    solutions2 = parse_solutions(text2)
    assert len(solutions2) >= 2, f"Expected at least 2 solutions, got {len(solutions2)}"
    print("  ✅ Multiple solutions parsed")

    # Test no solutions
    text3 = "No solutions exist for this problem"
    solutions3 = parse_solutions(text3)
    # May or may not find solutions depending on parsing
    print(f"  ✅ No-solution text processed (found {len(solutions3)} solutions)")

    return True


def test_normalize_solution():
    """Test solution normalization."""
    print("\nTesting normalize_solution...")

    # Test basic normalization
    assert normalize_solution("x=3") == normalize_solution("x = 3")
    print("  ✅ Spacing normalization works")

    # Test case normalization
    assert normalize_solution("X = 3") == normalize_solution("x = 3")
    print("  ✅ Case normalization works")

    # Test whitespace normalization
    assert normalize_solution("  x  =  3  ") == normalize_solution("x = 3")
    print("  ✅ Whitespace normalization works")

    return True


def test_mutual_exclusivity_metric():
    """Test mutual exclusivity scoring."""
    print("\nTesting mutual exclusivity metrics...")

    # Test good ME (non-overlapping cases)
    response_good = """
    Step 1: When x >= 0, solve x = 2x - 3, which gives x = 3
    Step 2: When x < 0, solve -x = 2x - 3, which gives x = -1
    """
    scores_good = compute_mutual_exclusivity(response_good, verbose=False)

    assert 'overall_me_score' in scores_good
    assert 'step_similarity_me_score' in scores_good
    assert 'case_overlap_me_score' in scores_good
    assert scores_good['n_steps'] >= 2
    print(f"  ✅ Good ME scored: {scores_good['overall_me_score']:.3f}")

    # Test poor ME (overlapping steps)
    response_poor = """
    Step 1: Solve x = 2x - 3
    Step 2: Solve x = 2x - 3
    Step 3: Solve x = 2x - 3
    """
    scores_poor = compute_mutual_exclusivity(response_poor, verbose=False)
    # Should have lower ME score due to repetition
    print(f"  ✅ Poor ME scored: {scores_poor['overall_me_score']:.3f}")

    # Test edge case: single step
    response_single = "Step 1: Solve the equation"
    scores_single = compute_mutual_exclusivity(response_single, verbose=False)
    assert scores_single['overall_me_score'] == 1.0, "Single step should have perfect ME"
    print("  ✅ Single step edge case handled")

    return True


def test_collective_exhaustiveness_metric():
    """Test collective exhaustiveness scoring."""
    print("\nTesting collective exhaustiveness metrics...")

    # Test good CE (all cases covered)
    response_good = """
    Case 1: When x >= 0, we get x = 3
    Case 2: When x < 0, we get x = -1
    The solutions are x = 3 and x = -1
    """
    problem_good = {
        "required_cases": ["x >= 0", "x < 0"],
        "ground_truth_solutions": ["x = 3", "x = -1"]
    }
    scores_good = compute_collective_exhaustiveness(
        response_good, problem_good, verbose=False
    )

    assert 'overall_ce_score' in scores_good
    assert 'case_enumeration_ce_score' in scores_good
    assert 'solution_coverage_ce_score' in scores_good
    print(f"  ✅ Good CE scored: {scores_good['overall_ce_score']:.3f}")

    # Test poor CE (missing cases/solutions)
    response_poor = """
    Case 1: When x >= 0, we get x = 3
    The solution is x = 3
    """
    problem_poor = {
        "required_cases": ["x >= 0", "x < 0"],
        "ground_truth_solutions": ["x = 3", "x = -1"]
    }
    scores_poor = compute_collective_exhaustiveness(
        response_poor, problem_poor, verbose=False
    )
    # Should have lower CE score due to missing case and solution
    print(f"  ✅ Poor CE scored: {scores_poor['overall_ce_score']:.3f}")
    assert len(scores_poor['missing_solutions']) > 0, "Should detect missing solutions"
    print(f"    Missing solutions detected: {scores_poor['missing_solutions']}")

    return True


def test_accuracy_metric():
    """Test accuracy scoring."""
    print("\nTesting accuracy metrics...")

    # Test exact match
    response_exact = "The solutions are x = 3 and x = -1"
    gt_exact = ["x = 3", "x = -1"]
    scores_exact = compute_accuracy(response_exact, gt_exact, verbose=False)

    assert 'exact_match' in scores_exact
    assert 'precision' in scores_exact
    assert 'recall' in scores_exact
    assert 'f1_score' in scores_exact
    print(f"  ✅ Exact match scored: F1={scores_exact['f1_score']:.3f}")

    # Test partial match
    response_partial = "The solution is x = 3"
    gt_partial = ["x = 3", "x = -1"]
    scores_partial = compute_accuracy(response_partial, gt_partial, verbose=False)
    assert scores_partial['exact_match'] == False, "Should not be exact match"
    assert scores_partial['precision'] > 0, "Should have some precision"
    assert scores_partial['recall'] < 1.0, "Should have incomplete recall"
    print(f"  ✅ Partial match scored: Precision={scores_partial['precision']:.3f}, Recall={scores_partial['recall']:.3f}")

    # Test wrong answer
    response_wrong = "The solution is x = 5"
    gt_wrong = ["x = 3"]
    scores_wrong = compute_accuracy(response_wrong, gt_wrong, verbose=False)
    assert scores_wrong['n_correct'] == 0, "Should have no correct solutions"
    print("  ✅ Wrong answer scored correctly")

    return True


def test_end_to_end_scenario():
    """Test a complete end-to-end scenario."""
    print("\nTesting end-to-end scenario...")

    # Simulate a complete model response
    response = """
    Let me solve |x - 3| = 5 using case analysis.

    Step 1: Identify the cases
    We need to consider when the expression inside is positive or negative.

    Step 2: Case 1 - When x - 3 >= 0 (i.e., x >= 3)
    In this case, |x - 3| = x - 3
    So: x - 3 = 5
    Therefore: x = 8

    Step 3: Case 2 - When x - 3 < 0 (i.e., x < 3)
    In this case, |x - 3| = -(x - 3) = -x + 3
    So: -x + 3 = 5
    Therefore: -x = 2
    Therefore: x = -2

    Step 4: Verify solutions
    For x = 8: |8 - 3| = |5| = 5 ✓
    For x = -2: |-2 - 3| = |-5| = 5 ✓

    Final answer: x = 8 or x = -2
    """

    problem = {
        "problem": "Solve |x - 3| = 5",
        "required_cases": ["x >= 3", "x < 3"],
        "ground_truth_solutions": ["x = 8", "x = -2"]
    }

    # Compute all metrics
    me_scores = compute_mutual_exclusivity(response, verbose=False)
    ce_scores = compute_collective_exhaustiveness(response, problem, verbose=False)
    acc_scores = compute_accuracy(response, problem["ground_truth_solutions"], verbose=False)

    print(f"  ME Score: {me_scores['overall_me_score']:.3f}")
    print(f"  CE Score: {ce_scores['overall_ce_score']:.3f}")
    print(f"  Accuracy: {acc_scores['f1_score']:.3f}")

    # All metrics should be high for this good response
    assert me_scores['overall_me_score'] > 0.5, "ME score should be reasonable"
    assert ce_scores['overall_ce_score'] > 0.5, "CE score should be reasonable"
    assert acc_scores['exact_match'] or acc_scores['f1_score'] > 0.8, "Should get good accuracy"

    print("  ✅ End-to-end scenario passed")

    return True


def main():
    """Run all Phase 3 metric tests."""
    print("=" * 70)
    print("PHASE 3 METRICS TEST")
    print("=" * 70)

    results = {}

    # Parser tests
    print("\n" + "=" * 70)
    print("PARSER TESTS")
    print("=" * 70)
    results["Parse Reasoning Steps"] = test_parse_reasoning_steps()
    results["Extract Case Conditions"] = test_extract_case_conditions()
    results["Count Overlapping Conditions"] = test_count_overlapping_conditions()
    results["Parse Solutions"] = test_parse_solutions()
    results["Normalize Solution"] = test_normalize_solution()

    # Metric tests
    print("\n" + "=" * 70)
    print("METRIC TESTS")
    print("=" * 70)
    results["Mutual Exclusivity Metric"] = test_mutual_exclusivity_metric()
    results["Collective Exhaustiveness Metric"] = test_collective_exhaustiveness_metric()
    results["Accuracy Metric"] = test_accuracy_metric()

    # End-to-end test
    print("\n" + "=" * 70)
    print("END-TO-END TEST")
    print("=" * 70)
    results["End-to-End Scenario"] = test_end_to_end_scenario()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:40} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL PHASE 3 METRIC TESTS PASSED!")
        print("\nPhase 3 Deliverables:")
        print("  ✅ Parsing utilities implemented")
        print("  ✅ Mutual exclusivity metrics implemented")
        print("  ✅ Collective exhaustiveness metrics implemented")
        print("  ✅ Accuracy metrics implemented")
        print("  ✅ All unit tests passing")
        print("\nNext steps:")
        print("  1. Proceed to Phase 4: Evaluation Pipeline")
        print("  2. Create evaluator.py to run metrics on full dataset")
        print("  3. Test with actual model outputs")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
