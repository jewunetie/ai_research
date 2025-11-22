"""
Unit tests for dataset validation.
"""

import json
import pytest
from pathlib import Path


# Get dataset path
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "math_case_analysis.json"


@pytest.fixture
def dataset():
    """Load the dataset."""
    with open(DATASET_PATH, 'r') as f:
        return json.load(f)


def test_dataset_exists():
    """Test that dataset file exists."""
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"


def test_dataset_is_valid_json():
    """Test that dataset is valid JSON."""
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    assert isinstance(data, list), "Dataset must be a list"


def test_dataset_count(dataset):
    """Test that dataset has correct number of problems."""
    expected_total = 15 + 15 + 10 + 10  # abs_val + piecewise + sign + range
    assert len(dataset) == expected_total, \
        f"Expected {expected_total} problems, found {len(dataset)}"


def test_problem_structure(dataset):
    """Test that each problem has required fields."""
    required_fields = {
        "id", "category", "problem", "required_cases",
        "ground_truth_solutions", "explanation"
    }

    for i, problem in enumerate(dataset):
        assert isinstance(problem, dict), f"Problem {i} must be a dict"
        problem_fields = set(problem.keys())
        assert required_fields.issubset(problem_fields), \
            f"Problem {i} missing fields: {required_fields - problem_fields}"


def test_problem_ids_unique(dataset):
    """Test that all problem IDs are unique."""
    ids = [p["id"] for p in dataset]
    assert len(ids) == len(set(ids)), "Problem IDs must be unique"


def test_categories_valid(dataset):
    """Test that all categories are valid."""
    valid_categories = {"absolute_value", "piecewise_functions",
                       "sign_analysis", "range_based"}

    for problem in dataset:
        assert problem["category"] in valid_categories, \
            f"Invalid category: {problem['category']}"


def test_category_distribution(dataset):
    """Test that problems are distributed correctly across categories."""
    expected = {
        "absolute_value": 15,
        "piecewise_functions": 15,
        "sign_analysis": 10,
        "range_based": 10
    }

    actual = {}
    for problem in dataset:
        cat = problem["category"]
        actual[cat] = actual.get(cat, 0) + 1

    assert actual == expected, \
        f"Category distribution mismatch.\nExpected: {expected}\nActual: {actual}"


def test_required_cases_not_empty(dataset):
    """Test that each problem has at least one required case."""
    for problem in dataset:
        assert isinstance(problem["required_cases"], list), \
            f"Problem {problem['id']}: required_cases must be a list"
        assert len(problem["required_cases"]) > 0, \
            f"Problem {problem['id']}: required_cases cannot be empty"


def test_solutions_not_empty(dataset):
    """Test that each problem has at least one solution."""
    for problem in dataset:
        assert isinstance(problem["ground_truth_solutions"], list), \
            f"Problem {problem['id']}: ground_truth_solutions must be a list"
        assert len(problem["ground_truth_solutions"]) > 0, \
            f"Problem {problem['id']}: ground_truth_solutions cannot be empty"


def test_problem_text_not_empty(dataset):
    """Test that problem text is not empty."""
    for problem in dataset:
        assert isinstance(problem["problem"], str), \
            f"Problem {problem['id']}: problem must be a string"
        assert problem["problem"].strip(), \
            f"Problem {problem['id']}: problem text cannot be empty"


def test_explanation_not_empty(dataset):
    """Test that explanation is not empty."""
    for problem in dataset:
        assert isinstance(problem["explanation"], str), \
            f"Problem {problem['id']}: explanation must be a string"
        assert problem["explanation"].strip(), \
            f"Problem {problem['id']}: explanation cannot be empty"


def test_id_format(dataset):
    """Test that problem IDs follow expected format."""
    for problem in dataset:
        id_str = problem["id"]
        assert isinstance(id_str, str), f"ID must be a string: {id_str}"
        assert "_" in id_str, f"ID should contain underscore: {id_str}"

        # Check ID matches category
        category_prefix = {
            "absolute_value": "abs_val",
            "piecewise_functions": "piecewise",
            "sign_analysis": "sign",
            "range_based": "range"
        }
        expected_prefix = category_prefix[problem["category"]]
        assert id_str.startswith(expected_prefix), \
            f"ID {id_str} should start with {expected_prefix} for category {problem['category']}"


def test_required_cases_are_strings(dataset):
    """Test that all required cases are strings."""
    for problem in dataset:
        for i, case in enumerate(problem["required_cases"]):
            assert isinstance(case, str), \
                f"Problem {problem['id']}: required_cases[{i}] must be a string"
            assert case.strip(), \
                f"Problem {problem['id']}: required_cases[{i}] cannot be empty"


def test_solutions_are_strings(dataset):
    """Test that all solutions are strings."""
    for problem in dataset:
        for i, sol in enumerate(problem["ground_truth_solutions"]):
            assert isinstance(sol, str), \
                f"Problem {problem['id']}: ground_truth_solutions[{i}] must be a string"
            assert sol.strip(), \
                f"Problem {problem['id']}: ground_truth_solutions[{i}] cannot be empty"


def test_absolute_value_problems(dataset):
    """Test specific properties of absolute value problems."""
    abs_val_problems = [p for p in dataset if p["category"] == "absolute_value"]
    assert len(abs_val_problems) == 15, "Should have 15 absolute value problems"

    # Most absolute value problems should have 2 cases
    for problem in abs_val_problems:
        # Just check they have at least 1 case (some might have different structures)
        assert len(problem["required_cases"]) >= 1, \
            f"Problem {problem['id']} should have at least 1 case"


def test_piecewise_problems(dataset):
    """Test specific properties of piecewise function problems."""
    piecewise_problems = [p for p in dataset if p["category"] == "piecewise_functions"]
    assert len(piecewise_problems) == 15, "Should have 15 piecewise problems"

    for problem in piecewise_problems:
        assert len(problem["required_cases"]) >= 2, \
            f"Piecewise problem {problem['id']} should have at least 2 cases"


def test_sign_analysis_problems(dataset):
    """Test specific properties of sign analysis problems."""
    sign_problems = [p for p in dataset if p["category"] == "sign_analysis"]
    assert len(sign_problems) == 10, "Should have 10 sign analysis problems"

    for problem in sign_problems:
        assert len(problem["required_cases"]) >= 2, \
            f"Sign analysis problem {problem['id']} should have at least 2 cases"


def test_range_based_problems(dataset):
    """Test specific properties of range-based problems."""
    range_problems = [p for p in dataset if p["category"] == "range_based"]
    assert len(range_problems) == 10, "Should have 10 range-based problems"

    for problem in range_problems:
        assert len(problem["required_cases"]) >= 1, \
            f"Range-based problem {problem['id']} should have at least 1 case"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
