"""Result aggregation utilities for combining and summarizing experiment results."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class ResultAggregator:
    """
    Aggregate and summarize experiment results.

    Combines results from multiple documents, conditions, and metrics
    into structured summaries and dataframes.
    """

    def __init__(self):
        """Initialize result aggregator."""
        self.results = []

    def add_result(self, result: Dict[str, Any]):
        """
        Add a single result to the aggregator.

        Args:
            result: Result dictionary
        """
        self.results.append(result)

    def add_results(self, results: List[Dict[str, Any]]):
        """
        Add multiple results to the aggregator.

        Args:
            results: List of result dictionaries
        """
        self.results.extend(results)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame(self.results)

    def aggregate_by_condition(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results by condition (variant or baseline).

        Args:
            metrics: List of metrics to aggregate (default: all numeric columns)

        Returns:
            Dictionary mapping condition -> metric -> aggregated value
        """
        if not self.results:
            return {}

        df = self.to_dataframe()

        if 'condition' not in df.columns:
            raise ValueError("Results must have 'condition' column for aggregation")

        # Determine metrics to aggregate
        if metrics is None:
            # Use all numeric columns except metadata
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if col not in ['doc_id', 'question_id']]

        # Aggregate by condition
        aggregated = {}
        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]

            condition_stats = {}
            for metric in metrics:
                if metric in condition_df.columns:
                    values = condition_df[metric].dropna()
                    if len(values) > 0:
                        condition_stats[metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'count': int(len(values)),
                        }

            aggregated[condition] = condition_stats

        return aggregated

    def aggregate_by_document(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results by document.

        Args:
            metrics: List of metrics to aggregate

        Returns:
            Dictionary mapping doc_id -> metric -> aggregated value
        """
        if not self.results:
            return {}

        df = self.to_dataframe()

        if 'doc_id' not in df.columns:
            raise ValueError("Results must have 'doc_id' column for aggregation")

        # Determine metrics
        if metrics is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if col not in ['doc_id', 'question_id']]

        # Aggregate by document
        aggregated = {}
        for doc_id in df['doc_id'].unique():
            doc_df = df[df['doc_id'] == doc_id]

            doc_stats = {}
            for metric in metrics:
                if metric in doc_df.columns:
                    values = doc_df[metric].dropna()
                    if len(values) > 0:
                        doc_stats[metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'count': int(len(values)),
                        }

            aggregated[doc_id] = doc_stats

        return aggregated

    def get_summary_statistics(
        self,
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get summary statistics for all metrics.

        Args:
            group_by: Column to group by (e.g., 'condition', 'doc_id')

        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            return pd.DataFrame()

        df = self.to_dataframe()

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metrics = [col for col in numeric_cols if col not in ['doc_id', 'question_id']]

        if group_by and group_by in df.columns:
            # Group by specified column
            summary = df.groupby(group_by)[metrics].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('count', 'count'),
            ])
        else:
            # Overall statistics
            summary = df[metrics].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('count', 'count'),
            ]).T

        return summary

    def compare_conditions(
        self,
        condition_a: str,
        condition_b: str,
        metric: str = 'f1',
    ) -> Dict[str, Any]:
        """
        Compare two conditions on a specific metric.

        Args:
            condition_a: First condition name
            condition_b: Second condition name
            metric: Metric to compare

        Returns:
            Dictionary with comparison statistics
        """
        df = self.to_dataframe()

        if 'condition' not in df.columns:
            raise ValueError("Results must have 'condition' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Get values for each condition
        values_a = df[df['condition'] == condition_a][metric].dropna()
        values_b = df[df['condition'] == condition_b][metric].dropna()

        if len(values_a) == 0 or len(values_b) == 0:
            return {
                'condition_a': condition_a,
                'condition_b': condition_b,
                'metric': metric,
                'error': 'No data for one or both conditions',
            }

        # Calculate statistics
        mean_a = values_a.mean()
        mean_b = values_b.mean()
        diff = mean_a - mean_b
        percent_diff = (diff / mean_b * 100) if mean_b != 0 else 0

        return {
            'condition_a': condition_a,
            'condition_b': condition_b,
            'metric': metric,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'std_a': float(values_a.std()),
            'std_b': float(values_b.std()),
            'diff': float(diff),
            'percent_diff': float(percent_diff),
            'n_a': int(len(values_a)),
            'n_b': int(len(values_b)),
        }

    def get_best_condition(
        self,
        metric: str = 'f1',
        higher_is_better: bool = True,
    ) -> Optional[str]:
        """
        Get the best-performing condition for a metric.

        Args:
            metric: Metric to evaluate
            higher_is_better: If True, higher values are better

        Returns:
            Name of best condition, or None if no data
        """
        aggregated = self.aggregate_by_condition(metrics=[metric])

        if not aggregated:
            return None

        # Get mean for each condition
        condition_means = {
            cond: stats[metric]['mean']
            for cond, stats in aggregated.items()
            if metric in stats
        }

        if not condition_means:
            return None

        if higher_is_better:
            return max(condition_means, key=condition_means.get)
        else:
            return min(condition_means, key=condition_means.get)

    def save_summary(
        self,
        output_path: Path,
        group_by: Optional[str] = 'condition',
    ):
        """
        Save summary statistics to file.

        Args:
            output_path: Path to save summary (JSON or CSV based on extension)
            group_by: Column to group by
        """
        output_path = Path(output_path)

        # Get summary
        summary = self.get_summary_statistics(group_by=group_by)

        # Save based on extension
        if output_path.suffix == '.json':
            # Convert to dict for JSON
            summary_dict = summary.to_dict()
            with open(output_path, 'w') as f:
                json.dump(summary_dict, f, indent=2)

        elif output_path.suffix == '.csv':
            summary.to_csv(output_path)

        else:
            raise ValueError(f"Unsupported file extension: {output_path.suffix}")

        print(f"✓ Summary saved to: {output_path}")

    def save_full_results(
        self,
        output_path: Path,
    ):
        """
        Save full results dataframe.

        Args:
            output_path: Path to save results (CSV or JSON)
        """
        output_path = Path(output_path)
        df = self.to_dataframe()

        if output_path.suffix == '.json':
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {output_path.suffix}")

        print(f"✓ Full results saved to: {output_path}")

    def clear(self):
        """Clear all stored results."""
        self.results.clear()

    def __len__(self) -> int:
        """Get number of stored results."""
        return len(self.results)

    def __repr__(self) -> str:
        """String representation."""
        return f"ResultAggregator(n_results={len(self.results)})"
