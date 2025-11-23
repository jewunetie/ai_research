#!/usr/bin/env python3
"""
Phase 7: Analyze and compare baseline vs MECE evaluation results.

This script:
1. Loads results from both conditions
2. Computes aggregate statistics
3. Performs statistical significance tests
4. Per-category analysis
5. Generates comparison tables and visualizations

Usage:
    # Analyze specific result files
    python scripts/analyze_results.py \
        --baseline results/baseline_results_20251123.json \
        --mece results/mece_results_20251123.json

    # Auto-find latest results in directory
    python scripts/analyze_results.py --results-dir results/

    # Generate plots (requires matplotlib)
    python scripts/analyze_results.py --results-dir results/ --plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def compute_aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all results."""
    stats = {
        'n_problems': len(results),
        'accuracy': {},
        'mutual_exclusivity': {},
        'collective_exhaustiveness': {},
        'latency': {},
        'steps': {}
    }

    # Accuracy metrics
    accuracies = [r['metrics']['accuracy'] for r in results]
    stats['accuracy'] = {
        'exact_match_rate': sum(1 for a in accuracies if a['exact_match']) / len(accuracies),
        'avg_f1': sum(a['f1_score'] for a in accuracies) / len(accuracies),
        'avg_precision': sum(a['precision'] for a in accuracies) / len(accuracies),
        'avg_recall': sum(a['recall'] for a in accuracies) / len(accuracies),
    }

    # ME metrics (if available)
    me_results = [
        r['metrics']['mutual_exclusivity']
        for r in results
        if 'mutual_exclusivity' in r['metrics'] and 'error' not in r['metrics']['mutual_exclusivity']
    ]

    if me_results:
        stats['mutual_exclusivity'] = {
            'n_available': len(me_results),
            'avg_overall': sum(m['overall_me_score'] for m in me_results) / len(me_results),
            'avg_similarity': sum(m['step_similarity_me_score'] for m in me_results) / len(me_results),
            'avg_overlap': sum(m['case_overlap_me_score'] for m in me_results) / len(me_results),
        }
    else:
        stats['mutual_exclusivity'] = {'n_available': 0, 'note': 'Not available'}

    # CE metrics (if available)
    ce_results = [
        r['metrics']['collective_exhaustiveness']
        for r in results
        if 'collective_exhaustiveness' in r['metrics'] and 'error' not in r['metrics']['collective_exhaustiveness']
    ]

    if ce_results:
        stats['collective_exhaustiveness'] = {
            'n_available': len(ce_results),
            'avg_overall': sum(c['overall_ce_score'] for c in ce_results) / len(ce_results),
            'avg_case_enum': sum(c['case_enumeration_ce_score'] for c in ce_results) / len(ce_results),
            'avg_solution_coverage': sum(c['solution_coverage_ce_score'] for c in ce_results) / len(ce_results),
        }
    else:
        stats['collective_exhaustiveness'] = {'n_available': 0, 'note': 'Not available'}

    # Latency
    latencies = [r['latency_ms'] for r in results]
    stats['latency'] = {
        'avg_ms': sum(latencies) / len(latencies),
        'min_ms': min(latencies),
        'max_ms': max(latencies),
    }

    # Steps
    steps = [r['n_steps'] for r in results]
    stats['steps'] = {
        'avg': sum(steps) / len(steps),
        'min': min(steps),
        'max': max(steps),
    }

    return stats


def compute_per_category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute statistics per problem category."""
    # Group by category
    by_category = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # Compute stats for each category
    category_stats = {}
    for cat, cat_results in by_category.items():
        category_stats[cat] = compute_aggregate_stats(cat_results)

    return category_stats


def compare_conditions(
    baseline_stats: Dict[str, Any],
    mece_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare baseline vs MECE statistics."""
    comparison = {}

    # Accuracy
    comparison['accuracy'] = {
        'baseline_f1': baseline_stats['accuracy']['avg_f1'],
        'mece_f1': mece_stats['accuracy']['avg_f1'],
        'delta_f1': mece_stats['accuracy']['avg_f1'] - baseline_stats['accuracy']['avg_f1'],
        'baseline_exact_match': baseline_stats['accuracy']['exact_match_rate'],
        'mece_exact_match': mece_stats['accuracy']['exact_match_rate'],
        'delta_exact_match': mece_stats['accuracy']['exact_match_rate'] - baseline_stats['accuracy']['exact_match_rate'],
    }

    # ME (if available)
    if baseline_stats['mutual_exclusivity']['n_available'] > 0 and mece_stats['mutual_exclusivity']['n_available'] > 0:
        comparison['mutual_exclusivity'] = {
            'baseline_overall': baseline_stats['mutual_exclusivity']['avg_overall'],
            'mece_overall': mece_stats['mutual_exclusivity']['avg_overall'],
            'delta_overall': mece_stats['mutual_exclusivity']['avg_overall'] - baseline_stats['mutual_exclusivity']['avg_overall'],
        }

    # CE (if available)
    if baseline_stats['collective_exhaustiveness']['n_available'] > 0 and mece_stats['collective_exhaustiveness']['n_available'] > 0:
        comparison['collective_exhaustiveness'] = {
            'baseline_overall': baseline_stats['collective_exhaustiveness']['avg_overall'],
            'mece_overall': mece_stats['collective_exhaustiveness']['avg_overall'],
            'delta_overall': mece_stats['collective_exhaustiveness']['avg_overall'] - baseline_stats['collective_exhaustiveness']['avg_overall'],
        }

    # Latency
    comparison['latency'] = {
        'baseline_avg_ms': baseline_stats['latency']['avg_ms'],
        'mece_avg_ms': mece_stats['latency']['avg_ms'],
        'delta_ms': mece_stats['latency']['avg_ms'] - baseline_stats['latency']['avg_ms'],
    }

    # Steps
    comparison['steps'] = {
        'baseline_avg': baseline_stats['steps']['avg'],
        'mece_avg': mece_stats['steps']['avg'],
        'delta': mece_stats['steps']['avg'] - baseline_stats['steps']['avg'],
    }

    return comparison


def statistical_significance_test(
    baseline_results: List[Dict[str, Any]],
    mece_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform statistical significance tests (if scipy available).

    Uses paired t-test for matched problem pairs.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {'note': 'scipy not available for significance testing'}

    # Match problems by ID
    baseline_by_id = {r['problem_id']: r for r in baseline_results}
    mece_by_id = {r['problem_id']: r for r in mece_results}

    common_ids = set(baseline_by_id.keys()) & set(mece_by_id.keys())

    if len(common_ids) < 2:
        return {'note': 'Need at least 2 matched problems for significance test'}

    # Paired accuracy F1 scores
    baseline_f1s = [baseline_by_id[pid]['metrics']['accuracy']['f1_score'] for pid in common_ids]
    mece_f1s = [mece_by_id[pid]['metrics']['accuracy']['f1_score'] for pid in common_ids]

    t_stat, p_value = scipy_stats.ttest_rel(mece_f1s, baseline_f1s)

    sig_tests = {
        'accuracy_f1': {
            'n_pairs': len(common_ids),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
        }
    }

    # ME scores (if available)
    baseline_me = [
        baseline_by_id[pid]['metrics']['mutual_exclusivity']['overall_me_score']
        for pid in common_ids
        if 'mutual_exclusivity' in baseline_by_id[pid]['metrics']
        and 'error' not in baseline_by_id[pid]['metrics']['mutual_exclusivity']
    ]
    mece_me = [
        mece_by_id[pid]['metrics']['mutual_exclusivity']['overall_me_score']
        for pid in common_ids
        if 'mutual_exclusivity' in mece_by_id[pid]['metrics']
        and 'error' not in mece_by_id[pid]['metrics']['mutual_exclusivity']
    ]

    if len(baseline_me) == len(mece_me) and len(baseline_me) >= 2:
        t_stat_me, p_value_me = scipy_stats.ttest_rel(mece_me, baseline_me)
        sig_tests['mutual_exclusivity'] = {
            'n_pairs': len(baseline_me),
            't_statistic': float(t_stat_me),
            'p_value': float(p_value_me),
            'significant_at_0.05': p_value_me < 0.05,
        }

    return sig_tests


def print_analysis(
    baseline_stats: Dict[str, Any],
    mece_stats: Dict[str, Any],
    comparison: Dict[str, Any],
    significance: Dict[str, Any]
):
    """Print analysis results to console."""
    print("\n" + "=" * 70)
    print("MECE vs BASELINE COMPARISON")
    print("=" * 70)

    # Accuracy
    print("\n📊 ACCURACY")
    print(f"  Baseline F1:     {comparison['accuracy']['baseline_f1']:.3f}")
    print(f"  MECE F1:         {comparison['accuracy']['mece_f1']:.3f}")
    print(f"  Δ (improvement): {comparison['accuracy']['delta_f1']:+.3f}")

    print(f"\n  Baseline Exact Match: {comparison['accuracy']['baseline_exact_match']:.1%}")
    print(f"  MECE Exact Match:     {comparison['accuracy']['mece_exact_match']:.1%}")
    print(f"  Δ (improvement):      {comparison['accuracy']['delta_exact_match']:+.1%}")

    # ME
    if 'mutual_exclusivity' in comparison:
        print("\n📊 MUTUAL EXCLUSIVITY (ME)")
        print(f"  Baseline ME:     {comparison['mutual_exclusivity']['baseline_overall']:.3f}")
        print(f"  MECE ME:         {comparison['mutual_exclusivity']['mece_overall']:.3f}")
        print(f"  Δ (improvement): {comparison['mutual_exclusivity']['delta_overall']:+.3f}")

    # CE
    if 'collective_exhaustiveness' in comparison:
        print("\n📊 COLLECTIVE EXHAUSTIVENESS (CE)")
        print(f"  Baseline CE:     {comparison['collective_exhaustiveness']['baseline_overall']:.3f}")
        print(f"  MECE CE:         {comparison['collective_exhaustiveness']['mece_overall']:.3f}")
        print(f"  Δ (improvement): {comparison['collective_exhaustiveness']['delta_overall']:+.3f}")

    # Steps
    print("\n📊 REASONING STEPS")
    print(f"  Baseline avg:    {comparison['steps']['baseline_avg']:.1f}")
    print(f"  MECE avg:        {comparison['steps']['mece_avg']:.1f}")
    print(f"  Δ (difference):  {comparison['steps']['delta']:+.1f}")

    # Latency
    print("\n⏱️  LATENCY")
    print(f"  Baseline avg:    {comparison['latency']['baseline_avg_ms']:.0f}ms")
    print(f"  MECE avg:        {comparison['latency']['mece_avg_ms']:.0f}ms")
    print(f"  Δ (difference):  {comparison['latency']['delta_ms']:+.0f}ms")

    # Statistical significance
    if 'note' not in significance:
        print("\n📈 STATISTICAL SIGNIFICANCE")
        acc_sig = significance.get('accuracy_f1', {})
        if acc_sig:
            print(f"\n  Accuracy F1 (paired t-test, n={acc_sig['n_pairs']}):")
            print(f"    t-statistic: {acc_sig['t_statistic']:.3f}")
            print(f"    p-value:     {acc_sig['p_value']:.4f}")
            if acc_sig['significant_at_0.05']:
                print(f"    ✅ Significant at p < 0.05")
            else:
                print(f"    ❌ Not significant at p < 0.05")

        if 'mutual_exclusivity' in significance:
            me_sig = significance['mutual_exclusivity']
            print(f"\n  ME Score (paired t-test, n={me_sig['n_pairs']}):")
            print(f"    t-statistic: {me_sig['t_statistic']:.3f}")
            print(f"    p-value:     {me_sig['p_value']:.4f}")
            if me_sig['significant_at_0.05']:
                print(f"    ✅ Significant at p < 0.05")
            else:
                print(f"    ❌ Not significant at p < 0.05")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze MECE evaluation results")

    parser.add_argument(
        '--baseline',
        type=Path,
        help='Path to baseline results JSON file'
    )

    parser.add_argument(
        '--mece',
        type=Path,
        help='Path to MECE results JSON file'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Directory containing result files (will auto-find latest)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for analysis results (JSON)'
    )

    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )

    args = parser.parse_args()

    # Find result files
    if args.baseline and args.mece:
        baseline_file = args.baseline
        mece_file = args.mece
    else:
        # Auto-find latest
        print(f"Auto-finding latest results in {args.results_dir}/...")

        baseline_files = sorted(glob.glob(str(args.results_dir / 'baseline_results_*.json')))
        mece_files = sorted(glob.glob(str(args.results_dir / 'mece_results_*.json')))

        if not baseline_files or not mece_files:
            print("❌ Could not find both baseline and MECE result files")
            print(f"   Looking in: {args.results_dir}")
            print(f"   Found {len(baseline_files)} baseline, {len(mece_files)} MECE files")
            return 1

        baseline_file = Path(baseline_files[-1])
        mece_file = Path(mece_files[-1])

        print(f"  Baseline: {baseline_file.name}")
        print(f"  MECE:     {mece_file.name}")

    # Load results
    print(f"\nLoading results...")
    baseline_results = load_results(baseline_file)
    mece_results = load_results(mece_file)
    print(f"  Baseline: {len(baseline_results)} problems")
    print(f"  MECE:     {len(mece_results)} problems")

    # Compute statistics
    print(f"\nComputing statistics...")
    baseline_stats = compute_aggregate_stats(baseline_results)
    mece_stats = compute_aggregate_stats(mece_results)

    # Compare
    comparison = compare_conditions(baseline_stats, mece_stats)

    # Statistical significance
    significance = statistical_significance_test(baseline_results, mece_results)

    # Print analysis
    print_analysis(baseline_stats, mece_stats, comparison, significance)

    # Per-category analysis
    print("\n" + "=" * 70)
    print("PER-CATEGORY ANALYSIS")
    print("=" * 70)

    baseline_by_cat = compute_per_category_stats(baseline_results)
    mece_by_cat = compute_per_category_stats(mece_results)

    for category in sorted(set(baseline_by_cat.keys()) | set(mece_by_cat.keys())):
        if category in baseline_by_cat and category in mece_by_cat:
            base_cat = baseline_by_cat[category]
            mece_cat = mece_by_cat[category]

            print(f"\n📂 {category.upper()} ({base_cat['n_problems']} problems)")
            print(f"   Baseline F1: {base_cat['accuracy']['avg_f1']:.3f}")
            print(f"   MECE F1:     {mece_cat['accuracy']['avg_f1']:.3f}")
            print(f"   Δ:           {mece_cat['accuracy']['avg_f1'] - base_cat['accuracy']['avg_f1']:+.3f}")

    # Save analysis (if requested)
    if args.output:
        analysis_data = {
            'baseline_stats': baseline_stats,
            'mece_stats': mece_stats,
            'comparison': comparison,
            'significance': significance,
            'baseline_by_category': baseline_by_cat,
            'mece_by_category': mece_by_cat,
        }

        with open(args.output, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\n✅ Analysis saved to: {args.output}")

    print("\n✅ Analysis complete!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
