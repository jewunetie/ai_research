#!/usr/bin/env python3
"""
Dataset creation and validation script for MECE Step by Step Reasoning.

This script validates the math_case_analysis.json dataset to ensure:
- Proper JSON format
- Required fields present
- Correct data types
- Valid problem categories
- Ground truth solutions are well-formed
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


# Valid categories
VALID_CATEGORIES = {
    "absolute_value",
    "piecewise_functions",
    "sign_analysis",
    "range_based"
}

# Expected problem counts per category
EXPECTED_COUNTS = {
    "absolute_value": 15,
    "piecewise_functions": 15,
    "sign_analysis": 10,
    "range_based": 10
}


def validate_problem(problem: Dict[str, Any], index: int) -> List[str]:
    """
    Validate a single problem.

    Args:
        problem: Problem dictionary
        index: Index in dataset for error reporting

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    required_fields = ["id", "category", "problem", "required_cases",
                      "ground_truth_solutions", "explanation"]

    for field in required_fields:
        if field not in problem:
            errors.append(f"Problem {index}: Missing required field '{field}'")

    if errors:  # If basic fields are missing, return early
        return errors

    # Validate ID format
    if not isinstance(problem["id"], str) or not problem["id"]:
        errors.append(f"Problem {index}: 'id' must be a non-empty string")

    # Validate category
    if problem["category"] not in VALID_CATEGORIES:
        errors.append(
            f"Problem {index} ({problem['id']}): "
            f"Invalid category '{problem['category']}'. "
            f"Must be one of {VALID_CATEGORIES}"
        )

    # Validate problem text
    if not isinstance(problem["problem"], str) or not problem["problem"].strip():
        errors.append(f"Problem {index} ({problem['id']}): 'problem' must be a non-empty string")

    # Validate required_cases
    if not isinstance(problem["required_cases"], list):
        errors.append(f"Problem {index} ({problem['id']}): 'required_cases' must be a list")
    elif len(problem["required_cases"]) == 0:
        errors.append(f"Problem {index} ({problem['id']}): 'required_cases' cannot be empty")
    else:
        for i, case in enumerate(problem["required_cases"]):
            if not isinstance(case, str) or not case.strip():
                errors.append(
                    f"Problem {index} ({problem['id']}): "
                    f"required_cases[{i}] must be a non-empty string"
                )

    # Validate ground_truth_solutions
    if not isinstance(problem["ground_truth_solutions"], list):
        errors.append(
            f"Problem {index} ({problem['id']}): "
            f"'ground_truth_solutions' must be a list"
        )
    elif len(problem["ground_truth_solutions"]) == 0:
        errors.append(
            f"Problem {index} ({problem['id']}): "
            f"'ground_truth_solutions' cannot be empty"
        )
    else:
        for i, sol in enumerate(problem["ground_truth_solutions"]):
            if not isinstance(sol, str) or not sol.strip():
                errors.append(
                    f"Problem {index} ({problem['id']}): "
                    f"ground_truth_solutions[{i}] must be a non-empty string"
                )

    # Validate explanation
    if not isinstance(problem["explanation"], str) or not problem["explanation"].strip():
        errors.append(
            f"Problem {index} ({problem['id']}): "
            f"'explanation' must be a non-empty string"
        )

    return errors


def validate_dataset(dataset_path: Path) -> bool:
    """
    Validate the entire dataset.

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        True if valid, False otherwise
    """
    print(f"Validating dataset: {dataset_path}")
    print("=" * 60)

    # Check file exists
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset file not found: {dataset_path}")
        return False

    # Load JSON
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format: {e}")
        return False

    # Check it's a list
    if not isinstance(dataset, list):
        print(f"❌ ERROR: Dataset must be a list of problems")
        return False

    # Check total count
    total_problems = len(dataset)
    expected_total = sum(EXPECTED_COUNTS.values())

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total problems: {total_problems}")
    print(f"   Expected: {expected_total}")

    if total_problems != expected_total:
        print(f"   ⚠️  WARNING: Expected {expected_total} problems, found {total_problems}")

    # Count by category
    category_counts = {cat: 0 for cat in VALID_CATEGORIES}
    problem_ids = set()
    all_errors = []

    for i, problem in enumerate(dataset):
        # Validate problem
        errors = validate_problem(problem, i)
        all_errors.extend(errors)

        # Count category (if valid)
        if "category" in problem and problem["category"] in VALID_CATEGORIES:
            category_counts[problem["category"]] += 1

        # Check for duplicate IDs
        if "id" in problem:
            if problem["id"] in problem_ids:
                all_errors.append(f"Duplicate problem ID: {problem['id']}")
            problem_ids.add(problem["id"])

    # Print category breakdown
    print(f"\n📚 Problems by Category:")
    for category in sorted(VALID_CATEGORIES):
        count = category_counts[category]
        expected = EXPECTED_COUNTS[category]
        status = "✓" if count == expected else "⚠️"
        print(f"   {status} {category:25} {count:2d} (expected {expected:2d})")

    # Print validation results
    print(f"\n🔍 Validation Results:")
    if all_errors:
        print(f"   ❌ Found {len(all_errors)} error(s):\n")
        for error in all_errors:
            print(f"      • {error}")
        return False
    else:
        print(f"   ✅ All {total_problems} problems are valid!")
        print(f"   ✅ All problem IDs are unique")
        print(f"   ✅ All categories are correctly distributed")
        return True


def main():
    """Main function."""
    # Get dataset path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / "data" / "math_case_analysis.json"

    # Validate
    is_valid = validate_dataset(dataset_path)

    # Print summary
    print("\n" + "=" * 60)
    if is_valid:
        print("✅ DATASET VALIDATION PASSED")
        print("\nDataset is ready for use!")
        sys.exit(0)
    else:
        print("❌ DATASET VALIDATION FAILED")
        print("\nPlease fix the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
