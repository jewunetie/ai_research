"""Statistical significance testing for experimental results."""

import numpy as np
from scipy import stats
from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class TTestResult:
    """Results from a t-test comparison."""

    statistic: float
    p_value: float
    mean_diff: float
    cohen_d: float  # Effect size
    significant: bool  # p < 0.05
    interpretation: str


@dataclass
class ANOVAResult:
    """Results from ANOVA test."""

    f_statistic: float
    p_value: float
    significant: bool
    num_groups: int
    interpretation: str


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size (standardized mean difference)

    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Human-readable interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> TTestResult:
    """Perform paired t-test to compare two related groups.

    Use this when comparing the same mechanism under different conditions,
    or two mechanisms on the same set of tasks.

    Args:
        group1: First set of measurements
        group2: Second set of measurements
        alpha: Significance level (default 0.05)

    Returns:
        TTestResult with test statistics and interpretation
    """
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(group1, group2)

    # Calculate effect size
    d = cohens_d(group1, group2)

    # Mean difference
    mean_diff = np.mean(group1) - np.mean(group2)

    # Interpret results
    significant = p_value < alpha
    effect_interpretation = interpret_cohens_d(d)

    if significant:
        if mean_diff > 0:
            interpretation = f"Group 1 significantly higher (p={p_value:.4f}, {effect_interpretation} effect)"
        else:
            interpretation = f"Group 2 significantly higher (p={p_value:.4f}, {effect_interpretation} effect)"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return TTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_diff=float(mean_diff),
        cohen_d=float(d),
        significant=significant,
        interpretation=interpretation
    )


def independent_t_test(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> TTestResult:
    """Perform independent samples t-test.

    Use this when comparing two independent groups (e.g., different mechanisms
    on independent task sets).

    Args:
        group1: First group of measurements
        group2: Second group of measurements
        alpha: Significance level

    Returns:
        TTestResult with test statistics and interpretation
    """
    # Perform independent t-test
    statistic, p_value = stats.ttest_ind(group1, group2)

    # Calculate effect size
    d = cohens_d(group1, group2)

    # Mean difference
    mean_diff = np.mean(group1) - np.mean(group2)

    # Interpret results
    significant = p_value < alpha
    effect_interpretation = interpret_cohens_d(d)

    if significant:
        if mean_diff > 0:
            interpretation = f"Group 1 significantly higher (p={p_value:.4f}, {effect_interpretation} effect)"
        else:
            interpretation = f"Group 2 significantly higher (p={p_value:.4f}, {effect_interpretation} effect)"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return TTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_diff=float(mean_diff),
        cohen_d=float(d),
        significant=significant,
        interpretation=interpretation
    )


def one_way_anova(*groups: np.ndarray, alpha: float = 0.05) -> ANOVAResult:
    """Perform one-way ANOVA to compare multiple groups.

    Use this to test if there are any significant differences between
    multiple mechanisms or conditions.

    Args:
        *groups: Variable number of groups to compare
        alpha: Significance level

    Returns:
        ANOVAResult with test statistics and interpretation
    """
    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)

    significant = p_value < alpha
    num_groups = len(groups)

    if significant:
        interpretation = (
            f"Significant difference found among {num_groups} groups "
            f"(F={f_statistic:.4f}, p={p_value:.4f})"
        )
    else:
        interpretation = (
            f"No significant difference among {num_groups} groups "
            f"(F={f_statistic:.4f}, p={p_value:.4f})"
        )

    return ANOVAResult(
        f_statistic=float(f_statistic),
        p_value=float(p_value),
        significant=significant,
        num_groups=num_groups,
        interpretation=interpretation
    )


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Bonferroni correction for multiple comparisons.

    When performing multiple t-tests, use this to adjust for the increased
    chance of false positives.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating which tests are significant after correction
    """
    adjusted_alpha = alpha / len(p_values)
    return [p < adjusted_alpha for p in p_values]


def compare_multiple_mechanisms(
    mechanism_results: dict[str, np.ndarray],
    alpha: float = 0.05
) -> dict[str, dict[str, TTestResult]]:
    """Perform pairwise comparisons between all mechanisms.

    Args:
        mechanism_results: Dict mapping mechanism name to array of results
        alpha: Significance level

    Returns:
        Nested dict with all pairwise comparison results
    """
    mechanism_names = list(mechanism_results.keys())
    comparisons = {}

    for i, mech1 in enumerate(mechanism_names):
        comparisons[mech1] = {}
        for mech2 in mechanism_names[i+1:]:
            result = paired_t_test(
                mechanism_results[mech1],
                mechanism_results[mech2],
                alpha=alpha
            )
            comparisons[mech1][mech2] = result

    return comparisons


def effect_size_matrix(
    mechanism_results: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Calculate Cohen's d for all pairwise comparisons.

    Args:
        mechanism_results: Dict mapping mechanism name to array of results

    Returns:
        Nested dict with Cohen's d for all pairs
    """
    mechanism_names = list(mechanism_results.keys())
    effect_sizes = {}

    for i, mech1 in enumerate(mechanism_names):
        effect_sizes[mech1] = {}
        for mech2 in mechanism_names:
            if mech1 == mech2:
                effect_sizes[mech1][mech2] = 0.0
            else:
                d = cohens_d(mechanism_results[mech1], mechanism_results[mech2])
                effect_sizes[mech1][mech2] = float(d)

    return effect_sizes


def print_comparison_summary(
    comparisons: dict[str, dict[str, TTestResult]],
    metric_name: str = "accuracy"
):
    """Print a formatted summary of all pairwise comparisons.

    Args:
        comparisons: Results from compare_multiple_mechanisms
        metric_name: Name of the metric being compared
    """
    print(f"\nPairwise Comparisons for {metric_name}:")
    print("=" * 80)

    for mech1, results in comparisons.items():
        for mech2, result in results.items():
            print(f"\n{mech1} vs {mech2}:")
            print(f"  Mean difference: {result.mean_diff:+.4f}")
            print(f"  Cohen's d: {result.cohen_d:.3f} ({interpret_cohens_d(result.cohen_d)} effect)")
            print(f"  p-value: {result.p_value:.4f}")
            print(f"  Significant: {'Yes' if result.significant else 'No'}")

    print("\n" + "=" * 80)
