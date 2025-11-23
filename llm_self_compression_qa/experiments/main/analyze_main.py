#!/usr/bin/env python3
"""
Analysis script for main experiment results.

Loads experiment results and computes detailed statistics and comparisons.

Usage:
    python experiments/main/analyze_main.py results/main/main_*.json [--full-report] [--figures]
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

# Import new analysis modules
try:
    from src.analysis.statistics import StatisticalAnalyzer, compare_conditions as compare_conditions_new
    from src.analysis.visualization import ExperimentVisualizer
    from src.analysis.report_generator import ReportGenerator
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError:
    ANALYSIS_MODULES_AVAILABLE = False
    print("⚠  Analysis modules not available. Some features will be disabled.")


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

    # Effect size (Cohen's d for paired data)
    # For paired data, use SD of differences, not pooled SD
    mean_diff = aligned_df["a"].mean() - aligned_df["b"].mean()
    diff = aligned_df["a"] - aligned_df["b"]
    std_diff = diff.std()
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

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


def print_analysis(results: Dict, results_file: Path):
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
    output_dir = results_file.parent
    df_file = output_dir / "analysis_dataframe.csv"
    df.to_csv(df_file, index=False)
    print(f"✓ Detailed results saved to: {df_file}")


def generate_advanced_analysis(
    results: Dict,
    df: pd.DataFrame,
    output_dir: Path,
    generate_figures: bool = False,
    generate_report: bool = False
):
    """
    Generate advanced analysis using new analysis modules.

    Args:
        results: Experiment results dictionary
        df: DataFrame with extracted metrics
        output_dir: Directory for output files
        generate_figures: Whether to generate figures
        generate_report: Whether to generate full report
    """
    if not ANALYSIS_MODULES_AVAILABLE:
        print("⚠  Advanced analysis modules not available. Skipping.")
        return

    print("=" * 80)
    print("Advanced Statistical Analysis")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05, confidence_level=0.95)

    # Perform all pairwise comparisons
    comparisons = []
    conditions = df["condition"].unique()

    for metric in ["exact_match", "f1", "semantic_similarity"]:
        if metric not in df.columns:
            continue

        metric_comparisons = analyzer.compare_all_conditions(
            df,
            metric=metric,
            condition_col="condition",
            paired=True
        )
        comparisons.extend(metric_comparisons)

    # Print comparison results
    print(f"Performed {len(comparisons)} pairwise comparisons")
    print()

    # Show significant comparisons
    significant = [c for c in comparisons if c.is_significant()]
    if significant:
        print(f"Significant comparisons (p < 0.05): {len(significant)}/{len(comparisons)}")
        print()
        for comp in significant[:10]:  # Show first 10
            print(str(comp))
            print()

    # Generate figures if requested
    figure_paths = None
    if generate_figures:
        print("=" * 80)
        print("Generating Visualization Figures")
        print("=" * 80)
        print()

        figures_dir = output_dir / "figures"
        visualizer = ExperimentVisualizer()

        # Determine baseline
        baseline = "full_context" if "full_context" in conditions else None

        # Generate all figures
        figure_paths = visualizer.create_full_report_figures(
            df,
            output_dir=figures_dir,
            metrics=["exact_match", "f1", "semantic_similarity"],
            baseline_condition=baseline
        )
        print()

        # Generate pairwise comparison figure
        if comparisons:
            path = figures_dir / "pairwise_comparisons.png"
            visualizer.plot_pairwise_comparison(
                comparisons,
                save_path=path,
                title="Statistical Comparisons Across All Metrics"
            )
            figure_paths["pairwise"] = path

    # Generate report if requested
    if generate_report:
        print("=" * 80)
        print("Generating Research Report")
        print("=" * 80)
        print()

        # Get summary statistics
        summary_stats = compute_summary_statistics(df)

        # Initialize report generator
        experiment_name = results.get("config", {}).get("name", "experiment")
        report_gen = ReportGenerator(experiment_name, output_dir)

        # Generate markdown report
        markdown = report_gen.generate_markdown_report(
            results=results,
            statistics=summary_stats,
            comparisons=comparisons,
            figure_paths=figure_paths
        )

        # Generate HTML report
        report_gen.generate_html_report(markdown)

        # Generate LaTeX table
        report_gen.generate_latex_table(summary_stats)

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze main experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_main.py results/main/main_20250101_120000.json

  # Generate figures only
  python analyze_main.py results/main/main_20250101_120000.json --figures

  # Generate full report with figures
  python analyze_main.py results/main/main_20250101_120000.json --full-report
        """
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to experiment results JSON file"
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate visualization figures"
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate full research report (markdown + HTML + figures)"
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1

    print(f"Loading results from: {args.results_file}")
    print()

    results = load_results(args.results_file)

    # Run basic analysis (always)
    print_analysis(results, args.results_file)
    print()

    # Run advanced analysis if requested
    if args.figures or args.full_report:
        df = extract_metrics_dataframe(results)
        output_dir = args.results_file.parent

        generate_advanced_analysis(
            results=results,
            df=df,
            output_dir=output_dir,
            generate_figures=args.figures or args.full_report,
            generate_report=args.full_report
        )

    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
