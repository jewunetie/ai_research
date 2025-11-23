#!/usr/bin/env python3
"""
Analysis script for main experiment results.

Loads experiment results and computes detailed statistics and comparisons.

Usage:
    python experiments/main/analyze_main.py results/main/main_*.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(results_file: Path) -> Dict:
    """Load experiment results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def extract_metrics_dataframe(results: Dict) -> pd.DataFrame:
    """
    Extract all metrics into a pandas DataFrame for analysis.

    Returns DataFrame with columns:
    - doc_id
    - condition (variant or baseline name)
    - question
    - reference_answer
    - answer
    - exact_match
    - f1
    - semantic_similarity
    """
    rows = []

    for doc in results["results"]:
        doc_id = doc["doc_id"]

        # Extract variant results
        for variant_name, variant_data in doc.get("variants", {}).items():
            for answer_data in variant_data["answers"]:
                rows.append({
                    "doc_id": doc_id,
                    "condition": variant_name,
                    "condition_type": "variant",
                    "question": answer_data["question"],
                    "reference_answer": answer_data["reference_answer"],
                    "answer": answer_data["answer"],
                    "exact_match": answer_data["metrics"]["exact_match"],
                    "f1": answer_data["metrics"]["f1"],
                    "semantic_similarity": answer_data["metrics"].get("semantic_similarity"),
                })

        # Extract baseline results
        for baseline_name, baseline_data in doc.get("baselines", {}).items():
            for answer_data in baseline_data["answers"]:
                rows.append({
                    "doc_id": doc_id,
                    "condition": baseline_name,
                    "condition_type": "baseline",
                    "question": answer_data["question"],
                    "reference_answer": answer_data["reference_answer"],
                    "answer": answer_data["answer"],
                    "exact_match": answer_data["metrics"]["exact_match"],
                    "f1": answer_data["metrics"]["f1"],
                    "semantic_similarity": answer_data["metrics"].get("semantic_similarity"),
                })

    return pd.DataFrame(rows)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by condition."""
    summary = df.groupby(["condition", "condition_type"]).agg({
        "exact_match": ["mean", "std", "count"],
        "f1": ["mean", "std"],
        "semantic_similarity": ["mean", "std"],
    }).round(4)

    return summary


def compare_conditions(df: pd.DataFrame, condition_a: str, condition_b: str):
    """
    Compare two conditions statistically.

    Uses paired t-test since same questions are answered in both conditions.
    """
    try:
        from scipy import stats
    except ImportError:
        print("⚠  scipy not installed. Install with 'pip install scipy' for statistical tests")
        return None

    # Get F1 scores for both conditions
    a_scores = df[df["condition"] == condition_a].groupby("doc_id")["f1"].mean()
    b_scores = df[df["condition"] == condition_b].groupby("doc_id")["f1"].mean()

    # Align by document ID
    aligned_df = pd.DataFrame({
        "a": a_scores,
        "b": b_scores
    }).dropna()

    if len(aligned_df) == 0:
        print(f"⚠  No overlapping documents for {condition_a} vs {condition_b}")
        return None

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(aligned_df["a"], aligned_df["b"])

    # Effect size (Cohen's d)
    mean_diff = aligned_df["a"].mean() - aligned_df["b"].mean()
    pooled_std = ((aligned_df["a"].std()**2 + aligned_df["b"].std()**2) / 2) ** 0.5
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "mean_a": aligned_df["a"].mean(),
        "mean_b": aligned_df["b"].mean(),
        "diff": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_pairs": len(aligned_df)
    }


def print_analysis(results: Dict):
    """Print comprehensive analysis of results."""
    print("=" * 80)
    print("Main Experiment Analysis")
    print("=" * 80)
    print()

    # Basic info
    config = results["config"]
    summary = results["summary"]

    print("Experiment Configuration:")
    print(f"  Name: {config['name']}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Documents: {summary['total_attempted']}")
    print(f"  Success rate: {summary['completion_rate']*100:.1f}%")
    print()

    # Cost summary
    if "cost_summary" in results:
        cost = results["cost_summary"]
        print("Cost Summary:")
        print(f"  API calls: {cost['calls_made']:,}")
        print(f"  Total tokens: {cost['total_input_tokens'] + cost['total_output_tokens']:,}")
        print(f"  Estimated cost: ${cost['estimated_cost']:.2f}")
        print()

    # Convert to DataFrame
    df = extract_metrics_dataframe(results)

    print(f"Total QA pairs: {len(df)}")
    print()

    # Summary statistics
    print("=" * 80)
    print("Summary Statistics by Condition")
    print("=" * 80)
    print()

    summary_stats = compute_summary_statistics(df)
    print(summary_stats)
    print()

    # Statistical comparisons
    print("=" * 80)
    print("Statistical Comparisons")
    print("=" * 80)
    print()

    conditions = df["condition"].unique()

    # Compare all variants to full_context baseline
    if "full_context" in conditions:
        for condition in conditions:
            if condition != "full_context":
                result = compare_conditions(df, condition, "full_context")
                if result:
                    print(f"{result['condition_a']} vs {result['condition_b']}:")
                    print(f"  Mean F1: {result['mean_a']:.3f} vs {result['mean_b']:.3f}")
                    print(f"  Difference: {result['diff']:.3f}")
                    print(f"  p-value: {result['p_value']:.4f} {'*' if result['p_value'] < 0.05 else ''}")
                    print(f"  Cohen's d: {result['cohens_d']:.3f}")
                    print(f"  N pairs: {result['n_pairs']}")
                    print()

    # Save DataFrame for further analysis
    output_dir = Path(results_file).parent
    df_file = output_dir / "analysis_dataframe.csv"
    df.to_csv(df_file, index=False)
    print(f"✓ Detailed results saved to: {df_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze main experiment results"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to experiment results JSON file"
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1

    print(f"Loading results from: {args.results_file}")
    print()

    results = load_results(args.results_file)
    print_analysis(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
