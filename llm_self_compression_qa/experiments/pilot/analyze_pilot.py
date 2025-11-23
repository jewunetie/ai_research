#!/usr/bin/env python3
"""
Analysis script for pilot experiment results.

Loads pilot results and provides quick analysis and visualization.

Usage:
    python experiments/pilot/analyze_pilot.py <results_dir>
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.statistics import StatisticalAnalyzer
from src.utils.logging_config import setup_logging


def analyze_pilot(results_dir: Path):
    """
    Analyze pilot experiment results.

    Args:
        results_dir: Directory containing pilot results
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print("="*80)
    print("PILOT EXPERIMENT ANALYSIS")
    print("="*80)
    print()

    # Load results
    full_results_path = results_dir / "full_results.csv"
    if not full_results_path.exists():
        print(f"Error: Results file not found: {full_results_path}")
        return 1

    print(f"Loading results from: {results_dir}")
    df = pd.read_csv(full_results_path)
    print(f"✓ Loaded {len(df)} results")
    print()

    # Basic info
    print("Dataset Info:")
    print(f"  Total QA pairs: {len(df)}")
    print(f"  Documents: {df['doc_id'].nunique()}")
    print(f"  Conditions: {', '.join(df['condition'].unique())}")
    print()

    # Summary statistics
    print("="*80)
    print("Summary Statistics by Condition")
    print("="*80)
    print()

    summary = df.groupby('condition').agg({
        'exact_match': ['mean', 'std', 'count'],
        'f1': ['mean', 'std'],
        'semantic_similarity': ['mean', 'std'],
    }).round(4)

    print(summary)
    print()

    # Condition comparisons
    print("="*80)
    print("Pairwise Comparisons")
    print("="*80)
    print()

    conditions = df['condition'].unique()

    # Compare compression to baselines
    if 'self_compression' in conditions and 'full_context' in conditions:
        print("Self-Compression vs Full Context:")
        comp_scores = df[df['condition'] == 'self_compression']['f1']
        full_scores = df[df['condition'] == 'full_context']['f1']

        print(f"  Self-Compression F1: {comp_scores.mean():.4f} ± {comp_scores.std():.4f}")
        print(f"  Full Context F1: {full_scores.mean():.4f} ± {full_scores.std():.4f}")
        print(f"  Difference: {(comp_scores.mean() - full_scores.mean()):.4f}")
        print(f"  Degradation: {((full_scores.mean() - comp_scores.mean()) / full_scores.mean() * 100):.1f}%")
        print()

    if 'self_compression' in conditions and 'no_context' in conditions:
        print("Self-Compression vs No Context:")
        comp_scores = df[df['condition'] == 'self_compression']['f1']
        no_ctx_scores = df[df['condition'] == 'no_context']['f1']

        print(f"  Self-Compression F1: {comp_scores.mean():.4f} ± {comp_scores.std():.4f}")
        print(f"  No Context F1: {no_ctx_scores.mean():.4f} ± {no_ctx_scores.std():.4f}")
        print(f"  Improvement: {(comp_scores.mean() - no_ctx_scores.mean()):.4f}")
        print(f"  Relative gain: {((comp_scores.mean() - no_ctx_scores.mean()) / no_ctx_scores.mean() * 100):.1f}%")
        print()

    # Statistical significance testing
    print("="*80)
    print("Statistical Tests")
    print("="*80)
    print()

    try:
        analyzer = StatisticalAnalyzer(alpha=0.05)

        if 'self_compression' in conditions and 'full_context' in conditions:
            # Paired comparison (same questions)
            comp_by_doc = df[df['condition'] == 'self_compression'].groupby('doc_id')['f1'].mean()
            full_by_doc = df[df['condition'] == 'full_context'].groupby('doc_id')['f1'].mean()

            # Align by doc_id
            aligned_docs = set(comp_by_doc.index) & set(full_by_doc.index)
            comp_aligned = comp_by_doc[list(aligned_docs)].sort_index()
            full_aligned = full_by_doc[list(aligned_docs)].sort_index()

            result = analyzer.compare_conditions_paired(
                comp_aligned.values,
                full_aligned.values,
                'self_compression',
                'full_context',
                'f1',
            )

            print("Self-Compression vs Full Context (paired t-test):")
            print(f"  t-statistic: {result.statistic:.3f}")
            print(f"  p-value: {result.p_value:.4f} {'***' if result.p_value < 0.001 else '**' if result.p_value < 0.01 else '*' if result.p_value < 0.05 else 'ns'}")
            print(f"  Cohen's d: {result.effect_size:.3f} ({result.interpret_effect_size()})")
            print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
            print()

    except Exception as e:
        print(f"Statistical testing failed: {e}")
        print()

    # Cost information
    cost_path = results_dir / "cost_summary.json"
    if cost_path.exists():
        print("="*80)
        print("Cost Summary")
        print("="*80)
        print()

        with open(cost_path) as f:
            cost = json.load(f)

        print(f"  API calls: {cost['calls_made']:,}")
        print(f"  Input tokens: {cost['total_input_tokens']:,}")
        print(f"  Output tokens: {cost['total_output_tokens']:,}")
        print(f"  Total tokens: {cost['total_input_tokens'] + cost['total_output_tokens']:,}")
        print(f"  Estimated cost: ${cost['estimated_cost']:.2f}")
        print()

    # Recommendations
    print("="*80)
    print("Recommendations for Main Experiment")
    print("="*80)
    print()

    # Check if results look reasonable
    if 'self_compression' in conditions and 'full_context' in conditions:
        comp_mean = df[df['condition'] == 'self_compression']['f1'].mean()
        full_mean = df[df['condition'] == 'full_context']['f1'].mean()

        degradation = (full_mean - comp_mean) / full_mean * 100

        if degradation < 10:
            print("✓ Performance degradation is minimal (<10%). Pipeline looks good!")
        elif degradation < 30:
            print("⚠ Moderate performance degradation (10-30%). Consider investigating compression quality.")
        else:
            print("⚠ High performance degradation (>30%). Review compression prompts and parameters.")

        print()

    print("Next steps:")
    print("1. Review compressed outputs manually to check quality")
    print("2. Check QA pairs for relevance and answerability")
    print("3. If results look good, proceed to main experiment with more documents")
    print("4. Consider adjusting compression token limits or prompts based on results")
    print()

    print("="*80)
    print("Analysis Complete")
    print("="*80)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze pilot experiment results")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing pilot results",
    )

    args = parser.parse_args()

    return analyze_pilot(args.results_dir)


if __name__ == "__main__":
    sys.exit(main())
