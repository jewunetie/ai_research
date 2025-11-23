"""Statistical analysis utilities for experiment results.

Provides comprehensive statistical testing and effect size calculations
for comparing compression variants and baselines.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Results from comparing two conditions."""

    condition_a: str
    condition_b: str
    metric: str

    # Sample statistics
    n: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_diff: float

    # Statistical test results
    test_type: str  # "paired_ttest", "independent_ttest", "wilcoxon", "mannwhitney"
    statistic: float
    p_value: float

    # Effect size
    effect_size_type: str  # "cohens_d", "hedges_g", "glass_delta"
    effect_size: float

    # Confidence intervals
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence_level: float = 0.95

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if difference is statistically significant."""
        return self.p_value < alpha

    def interpret_effect_size(self) -> str:
        """Interpret effect size magnitude (Cohen's conventions)."""
        abs_d = abs(self.effect_size)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def __str__(self) -> str:
        """String representation of comparison."""
        sig_marker = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"

        return (
            f"{self.condition_a} vs {self.condition_b} ({self.metric}):\n"
            f"  Mean difference: {self.mean_diff:.4f} ({self.mean_a:.4f} - {self.mean_b:.4f})\n"
            f"  {self.test_type}: t={self.statistic:.3f}, p={self.p_value:.4f} {sig_marker}\n"
            f"  Effect size ({self.effect_size_type}): {self.effect_size:.3f} ({self.interpret_effect_size()})\n"
            f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for experiment results.

    Supports:
    - Paired and independent t-tests
    - Non-parametric tests (Wilcoxon, Mann-Whitney)
    - Effect size calculations (Cohen's d, Hedges' g)
    - Confidence intervals
    - Multiple comparison corrections
    """

    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Significance level for hypothesis tests
            confidence_level: Confidence level for intervals
        """
        self.alpha = alpha
        self.confidence_level = confidence_level

    def compare_conditions_paired(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        condition_a: str,
        condition_b: str,
        metric: str,
        use_parametric: bool = True
    ) -> ComparisonResult:
        """
        Compare two conditions with paired data (same questions).

        Args:
            data_a: Scores for condition A
            data_b: Scores for condition B (paired with A)
            condition_a: Name of condition A
            condition_b: Name of condition B
            metric: Name of metric being compared
            use_parametric: Use t-test if True, Wilcoxon if False

        Returns:
            ComparisonResult with statistics and test results
        """
        # Ensure arrays
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)

        if len(data_a) != len(data_b):
            raise ValueError("Paired data must have same length")

        # Basic statistics
        n = len(data_a)
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        std_a = np.std(data_a, ddof=1)
        std_b = np.std(data_b, ddof=1)
        mean_diff = mean_a - mean_b

        # Statistical test
        if use_parametric:
            statistic, p_value = stats.ttest_rel(data_a, data_b)
            test_type = "paired_ttest"
        else:
            statistic, p_value = stats.wilcoxon(data_a, data_b)
            test_type = "wilcoxon"

        # Effect size (Cohen's d for paired data)
        diff = data_a - data_b
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Confidence interval for mean difference
        se_diff = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, df=n-1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        return ComparisonResult(
            condition_a=condition_a,
            condition_b=condition_b,
            metric=metric,
            n=n,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            mean_diff=mean_diff,
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size_type="cohens_d",
            effect_size=cohens_d,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level
        )

    def compare_conditions_independent(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        condition_a: str,
        condition_b: str,
        metric: str,
        use_parametric: bool = True,
        equal_variance: bool = False
    ) -> ComparisonResult:
        """
        Compare two conditions with independent data.

        Args:
            data_a: Scores for condition A
            data_b: Scores for condition B (independent from A)
            condition_a: Name of condition A
            condition_b: Name of condition B
            metric: Name of metric being compared
            use_parametric: Use t-test if True, Mann-Whitney if False
            equal_variance: Assume equal variance for t-test

        Returns:
            ComparisonResult with statistics and test results
        """
        # Ensure arrays
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)

        # Basic statistics
        n_a = len(data_a)
        n_b = len(data_b)
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        std_a = np.std(data_a, ddof=1)
        std_b = np.std(data_b, ddof=1)
        mean_diff = mean_a - mean_b

        # Statistical test
        if use_parametric:
            statistic, p_value = stats.ttest_ind(data_a, data_b, equal_var=equal_variance)
            test_type = "independent_ttest"
        else:
            statistic, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            test_type = "mannwhitney"

        # Effect size (Cohen's d for independent samples)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for mean difference
        se_diff = pooled_std * np.sqrt(1/n_a + 1/n_b)
        df = n_a + n_b - 2
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, df=df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        return ComparisonResult(
            condition_a=condition_a,
            condition_b=condition_b,
            metric=metric,
            n=n_a,  # Note: assumes n_a == n_b for simplicity
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            mean_diff=mean_diff,
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size_type="cohens_d",
            effect_size=cohens_d,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level
        )

    def compare_all_conditions(
        self,
        df: pd.DataFrame,
        metric: str,
        condition_col: str = "condition",
        value_col: str = None,
        paired: bool = True
    ) -> List[ComparisonResult]:
        """
        Compare all pairs of conditions.

        Args:
            df: DataFrame with results
            metric: Metric column to compare (or value_col if different)
            condition_col: Column containing condition names
            value_col: Column with values (defaults to metric)
            paired: Whether data is paired

        Returns:
            List of ComparisonResults for all pairwise comparisons
        """
        if value_col is None:
            value_col = metric

        conditions = df[condition_col].unique()
        results = []

        for i, cond_a in enumerate(conditions):
            for cond_b in conditions[i+1:]:
                data_a = df[df[condition_col] == cond_a][value_col].values
                data_b = df[df[condition_col] == cond_b][value_col].values

                if paired:
                    if len(data_a) != len(data_b):
                        print(f"Warning: Skipping {cond_a} vs {cond_b} - unequal lengths for paired test")
                        continue
                    result = self.compare_conditions_paired(
                        data_a, data_b, cond_a, cond_b, metric
                    )
                else:
                    result = self.compare_conditions_independent(
                        data_a, data_b, cond_a, cond_b, metric
                    )

                results.append(result)

        return results

    def bonferroni_correction(self, results: List[ComparisonResult]) -> List[ComparisonResult]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            results: List of comparison results

        Returns:
            Same list with adjusted p-values

        Note: Modifies p_value in place
        """
        n_comparisons = len(results)

        for result in results:
            result.p_value = min(1.0, result.p_value * n_comparisons)

        return results

    def summary_table(self, df: pd.DataFrame, group_col: str = "condition") -> pd.DataFrame:
        """
        Generate summary statistics table by condition.

        Args:
            df: DataFrame with results
            group_col: Column to group by

        Returns:
            DataFrame with mean, std, min, max, count for each metric
        """
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = df.groupby(group_col)[numeric_cols].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ])

        return summary


# Standalone utility functions

def compute_effect_size(
    data_a: np.ndarray,
    data_b: np.ndarray,
    method: str = "cohens_d",
    paired: bool = True
) -> float:
    """
    Compute effect size for comparison.

    Args:
        data_a: First group data
        data_b: Second group data
        method: "cohens_d", "hedges_g", or "glass_delta"
        paired: Whether data is paired

    Returns:
        Effect size value
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)

    mean_diff = np.mean(data_a) - np.mean(data_b)

    if method == "cohens_d":
        if paired:
            # For paired data, use SD of differences
            diff = data_a - data_b
            std = np.std(diff, ddof=1)
        else:
            # For independent data, use pooled SD
            n_a = len(data_a)
            n_b = len(data_b)
            std_a = np.std(data_a, ddof=1)
            std_b = np.std(data_b, ddof=1)
            std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        return mean_diff / std if std > 0 else 0.0

    elif method == "hedges_g":
        # Hedges' g is Cohen's d with correction for small samples
        d = compute_effect_size(data_a, data_b, method="cohens_d", paired=paired)
        n = len(data_a) + len(data_b)
        correction = 1 - (3 / (4 * n - 9))
        return d * correction

    elif method == "glass_delta":
        # Glass's delta uses only control group SD
        std = np.std(data_b, ddof=1)
        return mean_diff / std if std > 0 else 0.0

    else:
        raise ValueError(f"Unknown method: {method}")


def compare_conditions(
    df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    metric: str = "f1",
    condition_col: str = "condition",
    paired: bool = True
) -> ComparisonResult:
    """
    Convenience function to compare two conditions.

    Args:
        df: DataFrame with results
        condition_a: First condition name
        condition_b: Second condition name
        metric: Metric to compare
        condition_col: Column with condition names
        paired: Whether data is paired

    Returns:
        ComparisonResult
    """
    analyzer = StatisticalAnalyzer()

    data_a = df[df[condition_col] == condition_a][metric].values
    data_b = df[df[condition_col] == condition_b][metric].values

    if paired:
        return analyzer.compare_conditions_paired(
            data_a, data_b, condition_a, condition_b, metric
        )
    else:
        return analyzer.compare_conditions_independent(
            data_a, data_b, condition_a, condition_b, metric
        )
