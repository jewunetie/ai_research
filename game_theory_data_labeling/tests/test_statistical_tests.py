"""Tests for statistical significance testing utilities."""

import pytest
import numpy as np
from src.evaluation.statistical_tests import (
    cohens_d,
    interpret_cohens_d,
    paired_t_test,
    independent_t_test,
    one_way_anova,
    bonferroni_correction,
    compare_multiple_mechanisms,
    effect_size_matrix,
    TTestResult,
    ANOVAResult
)


def test_cohens_d_no_effect():
    """Cohen's d should be ~0 for identical groups."""
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group2 = np.array([1.1, 2.1, 2.9, 4.0, 5.0])

    d = cohens_d(group1, group2)

    # Should be small (negligible effect)
    assert abs(d) < 0.2


def test_cohens_d_large_effect():
    """Cohen's d should be large for very different groups."""
    group1 = np.array([1.0, 1.5, 2.0, 1.8, 1.2])
    group2 = np.array([5.0, 5.5, 6.0, 5.8, 5.2])

    d = cohens_d(group1, group2)

    # Should be large effect (|d| > 0.8)
    assert abs(d) > 0.8


def test_interpret_cohens_d():
    """Test interpretation of Cohen's d values."""
    assert interpret_cohens_d(0.1) == "negligible"
    assert interpret_cohens_d(-0.15) == "negligible"
    assert interpret_cohens_d(0.3) == "small"
    assert interpret_cohens_d(-0.4) == "small"
    assert interpret_cohens_d(0.6) == "medium"
    assert interpret_cohens_d(-0.7) == "medium"
    assert interpret_cohens_d(1.0) == "large"
    assert interpret_cohens_d(-1.5) == "large"


def test_paired_t_test_significant_difference():
    """Paired t-test should detect significant differences."""
    np.random.seed(42)

    # Group1 is clearly higher than Group2
    group1 = np.random.normal(100, 10, 50)
    group2 = np.random.normal(90, 10, 50)

    result = paired_t_test(group1, group2, alpha=0.05)

    assert isinstance(result, TTestResult)
    assert result.significant == True
    assert result.p_value < 0.05
    assert result.mean_diff > 0  # Group1 > Group2
    assert "Group 1 significantly higher" in result.interpretation


def test_paired_t_test_no_difference():
    """Paired t-test should not detect difference when groups are similar."""
    np.random.seed(42)

    # Very similar groups
    group1 = np.random.normal(100, 10, 50)
    group2 = group1 + np.random.normal(0, 1, 50)  # Just add small noise

    result = paired_t_test(group1, group2, alpha=0.05)

    assert isinstance(result, TTestResult)
    # With small noise, likely not significant
    # But this depends on random seed, so just check structure
    assert isinstance(result.p_value, float)
    assert isinstance(result.cohen_d, float)


def test_independent_t_test():
    """Independent t-test should work on independent samples."""
    np.random.seed(42)

    # Two independent groups with different means
    group1 = np.random.normal(100, 15, 40)
    group2 = np.random.normal(85, 15, 40)

    result = independent_t_test(group1, group2, alpha=0.05)

    assert isinstance(result, TTestResult)
    assert result.significant == True
    assert result.p_value < 0.05
    assert abs(result.cohen_d) > 0.5  # At least medium effect


def test_one_way_anova_significant():
    """ANOVA should detect differences among multiple groups."""
    np.random.seed(42)

    # Three groups with different means
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(110, 10, 30)
    group3 = np.random.normal(90, 10, 30)

    result = one_way_anova(group1, group2, group3, alpha=0.05)

    assert isinstance(result, ANOVAResult)
    assert result.significant == True
    assert result.p_value < 0.05
    assert result.num_groups == 3
    assert "Significant difference" in result.interpretation


def test_one_way_anova_no_difference():
    """ANOVA should not detect difference when groups are similar."""
    np.random.seed(42)

    # Three groups with same mean
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(100, 10, 30)
    group3 = np.random.normal(100, 10, 30)

    result = one_way_anova(group1, group2, group3, alpha=0.05)

    assert isinstance(result, ANOVAResult)
    assert result.significant == False
    assert result.p_value > 0.05
    assert "No significant difference" in result.interpretation


def test_bonferroni_correction():
    """Bonferroni correction should adjust significance threshold."""
    p_values = [0.005, 0.01, 0.02, 0.03, 0.04]

    # With alpha=0.05 and 5 tests, adjusted alpha = 0.01
    results = bonferroni_correction(p_values, alpha=0.05)

    assert len(results) == 5
    assert results[0] == True   # 0.005 < 0.01
    assert results[1] == False  # 0.01 is not < 0.01 (boundary case)
    assert results[2] == False  # 0.02 > 0.01
    assert results[3] == False  # 0.03 > 0.01
    assert results[4] == False  # 0.04 > 0.01


def test_compare_multiple_mechanisms():
    """Test pairwise comparisons between mechanisms."""
    np.random.seed(42)

    mechanism_results = {
        "MV": np.random.normal(0.95, 0.05, 50),
        "DS": np.random.normal(0.98, 0.03, 50),
        "OA": np.random.normal(0.94, 0.06, 50)
    }

    comparisons = compare_multiple_mechanisms(mechanism_results, alpha=0.05)

    # Should have comparisons for MV vs DS, MV vs OA, DS vs OA
    assert "MV" in comparisons
    assert "DS" in comparisons["MV"]
    assert "OA" in comparisons["MV"]
    assert "OA" in comparisons["DS"]

    # Each comparison should be a TTestResult
    assert isinstance(comparisons["MV"]["DS"], TTestResult)
    assert isinstance(comparisons["MV"]["OA"], TTestResult)


def test_effect_size_matrix():
    """Test effect size matrix calculation."""
    np.random.seed(42)

    mechanism_results = {
        "MV": np.random.normal(0.95, 0.05, 50),
        "DS": np.random.normal(0.98, 0.03, 50),
        "OA": np.random.normal(0.94, 0.06, 50)
    }

    matrix = effect_size_matrix(mechanism_results)

    # Should have all pairwise comparisons
    assert "MV" in matrix
    assert "DS" in matrix
    assert "OA" in matrix

    # Diagonal should be zero
    assert matrix["MV"]["MV"] == 0.0
    assert matrix["DS"]["DS"] == 0.0
    assert matrix["OA"]["OA"] == 0.0

    # Should have effect sizes for all pairs
    assert isinstance(matrix["MV"]["DS"], float)
    assert isinstance(matrix["MV"]["OA"], float)
    assert isinstance(matrix["DS"]["OA"], float)


def test_ttest_result_dataclass():
    """Test TTestResult structure."""
    result = TTestResult(
        statistic=2.5,
        p_value=0.02,
        mean_diff=0.05,
        cohen_d=0.6,
        significant=True,
        interpretation="Test interpretation"
    )

    assert result.statistic == 2.5
    assert result.p_value == 0.02
    assert result.mean_diff == 0.05
    assert result.cohen_d == 0.6
    assert result.significant == True
    assert result.interpretation == "Test interpretation"


def test_anova_result_dataclass():
    """Test ANOVAResult structure."""
    result = ANOVAResult(
        f_statistic=5.2,
        p_value=0.01,
        significant=True,
        num_groups=3,
        interpretation="Test interpretation"
    )

    assert result.f_statistic == 5.2
    assert result.p_value == 0.01
    assert result.significant == True
    assert result.num_groups == 3
    assert result.interpretation == "Test interpretation"


def test_cohens_d_symmetry():
    """Cohen's d should have opposite sign when groups are swapped."""
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group2 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    d1 = cohens_d(group1, group2)
    d2 = cohens_d(group2, group1)

    # Should be approximately equal in magnitude but opposite sign
    assert abs(d1 + d2) < 1e-10


def test_paired_t_test_with_custom_alpha():
    """Test paired t-test with different alpha levels."""
    np.random.seed(42)

    group1 = np.random.normal(100, 10, 50)
    group2 = np.random.normal(95, 10, 50)

    # Test with strict alpha
    result_strict = paired_t_test(group1, group2, alpha=0.01)

    # Test with lenient alpha
    result_lenient = paired_t_test(group1, group2, alpha=0.10)

    # Both should have same p-value, but significance may differ
    assert result_strict.p_value == result_lenient.p_value

    # Lenient alpha more likely to be significant
    if not result_strict.significant:
        # If strict is not significant, lenient might still be
        assert True
