#!/usr/bin/env python3
"""
Phase 8: Generate documentation from evaluation results.

This script creates a comprehensive RESULTS.md file documenting:
1. Executive summary
2. Methodology
3. Key findings
4. Detailed metrics
5. Per-category analysis
6. Example responses (best/worst)
7. Limitations
8. Future work

Usage:
    # Generate from latest results
    python scripts/generate_documentation.py

    # Generate from specific analysis file
    python scripts/generate_documentation.py --analysis analysis_results.json

    # Specify output file
    python scripts/generate_documentation.py --output RESULTS.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_analysis(analysis_file: Path) -> dict:
    """Load analysis results from JSON."""
    with open(analysis_file) as f:
        return json.load(f)


def load_results_files(baseline_file: Path, mece_file: Path) -> tuple:
    """Load raw result files."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(mece_file) as f:
        mece = json.load(f)
    return baseline, mece


def find_interesting_examples(
    baseline_results: list,
    mece_results: list
) -> dict:
    """Find interesting example responses to highlight."""
    # Match by problem ID
    baseline_by_id = {r['problem_id']: r for r in baseline_results}
    mece_by_id = {r['problem_id']: r for r in mece_results}

    common_ids = set(baseline_by_id.keys()) & set(mece_by_id.keys())

    examples = {
        'best_mece': None,
        'best_baseline': None,
        'biggest_improvement': None,
        'biggest_decline': None,
    }

    # Find biggest improvement/decline
    max_improvement = -999
    max_decline = 999

    for pid in common_ids:
        baseline_f1 = baseline_by_id[pid]['metrics']['accuracy']['f1_score']
        mece_f1 = mece_by_id[pid]['metrics']['accuracy']['f1_score']
        delta = mece_f1 - baseline_f1

        if delta > max_improvement:
            max_improvement = delta
            examples['biggest_improvement'] = {
                'problem_id': pid,
                'baseline_f1': baseline_f1,
                'mece_f1': mece_f1,
                'delta': delta,
                'baseline_response': baseline_by_id[pid]['response'],
                'mece_response': mece_by_id[pid]['response'],
            }

        if delta < max_decline:
            max_decline = delta
            examples['biggest_decline'] = {
                'problem_id': pid,
                'baseline_f1': baseline_f1,
                'mece_f1': mece_f1,
                'delta': delta,
                'baseline_response': baseline_by_id[pid]['response'],
                'mece_response': mece_by_id[pid]['response'],
            }

    # Find best overall
    mece_sorted = sorted(mece_results, key=lambda r: r['metrics']['accuracy']['f1_score'], reverse=True)
    baseline_sorted = sorted(baseline_results, key=lambda r: r['metrics']['accuracy']['f1_score'], reverse=True)

    if mece_sorted:
        examples['best_mece'] = {
            'problem_id': mece_sorted[0]['problem_id'],
            'f1': mece_sorted[0]['metrics']['accuracy']['f1_score'],
            'response': mece_sorted[0]['response'],
        }

    if baseline_sorted:
        examples['best_baseline'] = {
            'problem_id': baseline_sorted[0]['problem_id'],
            'f1': baseline_sorted[0]['metrics']['accuracy']['f1_score'],
            'response': baseline_sorted[0]['response'],
        }

    return examples


def generate_markdown(
    analysis: dict,
    baseline_results: list,
    mece_results: list
) -> str:
    """Generate comprehensive RESULTS.md document."""

    md = []

    # Header
    md.append("# MECE Step-by-Step Reasoning: Evaluation Results\n")
    md.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
    md.append(f"**Model**: Qwen3-0.6B-Instruct\n")
    md.append(f"**Problems**: {len(baseline_results)} math case analysis problems\n")
    md.append("---\n")

    # Executive Summary
    md.append("## Executive Summary\n")

    comparison = analysis['comparison']
    accuracy_delta = comparison['accuracy']['delta_f1']
    baseline_f1 = comparison['accuracy']['baseline_f1']
    mece_f1 = comparison['accuracy']['mece_f1']

    if accuracy_delta > 0.05:
        verdict = f"**MECE prompting IMPROVED accuracy** by {accuracy_delta:+.1%}"
    elif accuracy_delta < -0.05:
        verdict = f"**MECE prompting DECREASED accuracy** by {accuracy_delta:+.1%}"
    else:
        verdict = f"**MECE prompting had MINIMAL IMPACT** on accuracy ({accuracy_delta:+.1%})"

    md.append(f"{verdict}\n\n")

    md.append("**Key Findings:**\n")
    md.append(f"- Baseline Accuracy (F1): {baseline_f1:.1%}\n")
    md.append(f"- MECE Accuracy (F1): {mece_f1:.1%}\n")
    md.append(f"- Difference: {accuracy_delta:+.1%}\n")

    if 'mutual_exclusivity' in comparison:
        me_delta = comparison['mutual_exclusivity']['delta_overall']
        md.append(f"- ME Score Δ: {me_delta:+.3f}\n")

    if 'collective_exhaustiveness' in comparison:
        ce_delta = comparison['collective_exhaustiveness']['delta_overall']
        md.append(f"- CE Score Δ: {ce_delta:+.3f}\n")

    md.append("\n---\n")

    # Methodology
    md.append("## Methodology\n\n")
    md.append("### Conditions\n\n")
    md.append("**Baseline**: Standard chain-of-thought prompting with thinking mode enabled\n\n")
    md.append("**MECE**: Structured prompting explicitly requesting Mutually Exclusive, Collectively Exhaustive decomposition\n\n")

    md.append("### Metrics\n\n")
    md.append("- **Accuracy (F1)**: Harmonic mean of precision and recall for solution correctness\n")
    md.append("- **ME Score**: Mutual exclusivity measured via embedding similarity and condition overlap (0-1, higher = better)\n")
    md.append("- **CE Score**: Collective exhaustiveness measured via case enumeration and solution coverage (0-1, higher = better)\n\n")

    md.append("---\n")

    # Detailed Results
    md.append("## Detailed Results\n\n")

    md.append("### Accuracy\n\n")
    md.append("| Metric | Baseline | MECE | Δ |\n")
    md.append("|--------|----------|------|---|\n")
    md.append(f"| F1 Score | {baseline_f1:.3f} | {mece_f1:.3f} | {accuracy_delta:+.3f} |\n")
    md.append(f"| Exact Match Rate | {comparison['accuracy']['baseline_exact_match']:.1%} | {comparison['accuracy']['mece_exact_match']:.1%} | {comparison['accuracy']['delta_exact_match']:+.1%} |\n")
    md.append("\n")

    # ME/CE if available
    if 'mutual_exclusivity' in comparison:
        md.append("### Mutual Exclusivity (ME)\n\n")
        md.append("| Metric | Baseline | MECE | Δ |\n")
        md.append("|--------|----------|------|---|\n")
        md.append(f"| Overall ME Score | {comparison['mutual_exclusivity']['baseline_overall']:.3f} | {comparison['mutual_exclusivity']['mece_overall']:.3f} | {comparison['mutual_exclusivity']['delta_overall']:+.3f} |\n")
        md.append("\n")

    if 'collective_exhaustiveness' in comparison:
        md.append("### Collective Exhaustiveness (CE)\n\n")
        md.append("| Metric | Baseline | MECE | Δ |\n")
        md.append("|--------|----------|------|---|\n")
        md.append(f"| Overall CE Score | {comparison['collective_exhaustiveness']['baseline_overall']:.3f} | {comparison['collective_exhaustiveness']['mece_overall']:.3f} | {comparison['collective_exhaustiveness']['delta_overall']:+.3f} |\n")
        md.append("\n")

    # Statistical Significance
    if 'note' not in analysis['significance']:
        md.append("### Statistical Significance\n\n")

        acc_sig = analysis['significance'].get('accuracy_f1', {})
        if acc_sig:
            md.append(f"**Accuracy F1** (paired t-test, n={acc_sig['n_pairs']}):\n")
            md.append(f"- t-statistic: {acc_sig['t_statistic']:.3f}\n")
            md.append(f"- p-value: {acc_sig['p_value']:.4f}\n")
            if acc_sig['significant_at_0.05']:
                md.append(f"- **Result**: ✅ Significant at p < 0.05\n")
            else:
                md.append(f"- **Result**: ❌ Not significant at p < 0.05\n")
            md.append("\n")

    # Per-Category Analysis
    md.append("---\n\n")
    md.append("## Per-Category Analysis\n\n")

    baseline_by_cat = analysis['baseline_by_category']
    mece_by_cat = analysis['mece_by_category']

    md.append("| Category | n | Baseline F1 | MECE F1 | Δ |\n")
    md.append("|----------|---|-------------|---------|---|\n")

    for category in sorted(baseline_by_cat.keys()):
        if category in mece_by_cat:
            base_cat = baseline_by_cat[category]
            mece_cat = mece_by_cat[category]
            n = base_cat['n_problems']
            base_f1 = base_cat['accuracy']['avg_f1']
            mece_f1 = mece_cat['accuracy']['avg_f1']
            delta = mece_f1 - base_f1

            md.append(f"| {category} | {n} | {base_f1:.3f} | {mece_f1:.3f} | {delta:+.3f} |\n")

    md.append("\n---\n\n")

    # Example Responses
    md.append("## Example Responses\n\n")

    examples = find_interesting_examples(baseline_results, mece_results)

    if examples['biggest_improvement'] and examples['biggest_improvement']['delta'] > 0:
        ex = examples['biggest_improvement']
        md.append(f"### Biggest Improvement: {ex['problem_id']}\n\n")
        md.append(f"**Baseline F1**: {ex['baseline_f1']:.3f} → **MECE F1**: {ex['mece_f1']:.3f} (Δ: {ex['delta']:+.3f})\n\n")
        md.append("**MECE Response (excerpt)**:\n")
        md.append(f"```\n{ex['mece_response'][:500]}...\n```\n\n")

    if examples['biggest_decline'] and examples['biggest_decline']['delta'] < 0:
        ex = examples['biggest_decline']
        md.append(f"### Biggest Decline: {ex['problem_id']}\n\n")
        md.append(f"**Baseline F1**: {ex['baseline_f1']:.3f} → **MECE F1**: {ex['mece_f1']:.3f} (Δ: {ex['delta']:+.3f})\n\n")
        md.append("**Baseline Response (excerpt)**:\n")
        md.append(f"```\n{ex['baseline_response'][:500]}...\n```\n\n")

    md.append("---\n\n")

    # Limitations
    md.append("## Limitations\n\n")
    md.append("1. **Small Model**: Qwen3-0.6B-Instruct is a very small model; results may differ with larger models\n")
    md.append("2. **Limited Dataset**: Only 50 problems in math domain; broader evaluation needed\n")
    md.append("3. **Single MECE Version**: Only tested MECE v1; v2 and v3 may perform differently\n")
    md.append("4. **Heuristic Metrics**: ME/CE metrics are approximations, not perfect measures\n")
    md.append("5. **No Few-Shot**: Both conditions used zero-shot prompting; few-shot may improve results\n\n")

    md.append("---\n\n")

    # Future Work
    md.append("## Future Work\n\n")
    md.append("1. **Larger Models**: Test on Qwen3-7B, 14B, or other capable models\n")
    md.append("2. **Broader Domains**: Evaluate on non-math reasoning tasks\n")
    md.append("3. **MECE Variants**: Compare v1, v2, v3 and ablate components\n")
    md.append("4. **Few-Shot Learning**: Add MECE examples to prompts\n")
    md.append("5. **Human Evaluation**: Validate computational metrics with human judges\n")
    md.append("6. **Error Analysis**: Deep dive into failure modes\n\n")

    md.append("---\n\n")

    # Conclusion
    md.append("## Conclusion\n\n")

    if accuracy_delta > 0.05:
        md.append(f"MECE prompting showed **positive results** on this evaluation, improving accuracy by {accuracy_delta:+.1%}. ")
    elif accuracy_delta < -0.05:
        md.append(f"MECE prompting showed **negative results** on this evaluation, decreasing accuracy by {accuracy_delta:+.1%}. ")
    else:
        md.append(f"MECE prompting showed **neutral results** on this evaluation, with minimal impact on accuracy ({accuracy_delta:+.1%}). ")

    if 'mutual_exclusivity' in comparison:
        me_delta = comparison['mutual_exclusivity']['delta_overall']
        if me_delta > 0.1:
            md.append(f"The ME score improvement ({me_delta:+.3f}) suggests MECE prompting did encourage more distinct reasoning steps. ")

    if 'collective_exhaustiveness' in comparison:
        ce_delta = comparison['collective_exhaustiveness']['delta_overall']
        if ce_delta > 0.1:
            md.append(f"The CE score improvement ({ce_delta:+.3f}) suggests better case coverage. ")

    md.append("\n\nFurther research with larger models and broader evaluation is recommended to validate these findings.\n")

    return ''.join(md)


def main():
    parser = argparse.ArgumentParser(description="Generate documentation from evaluation results")

    parser.add_argument(
        '--analysis',
        type=Path,
        help='Path to analysis JSON file (from analyze_results.py)'
    )

    parser.add_argument(
        '--baseline',
        type=Path,
        help='Path to baseline results JSON'
    )

    parser.add_argument(
        '--mece',
        type=Path,
        help='Path to MECE results JSON'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Directory to search for result files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('RESULTS.md'),
        help='Output markdown file (default: RESULTS.md)'
    )

    args = parser.parse_args()

    # Find files
    if args.analysis:
        print(f"Loading analysis from: {args.analysis}")
        analysis = load_analysis(args.analysis)

        # Still need raw results for examples
        if not (args.baseline and args.mece):
            print("Need --baseline and --mece with --analysis for complete documentation")
            return 1

        baseline_results, mece_results = load_results_files(args.baseline, args.mece)

    else:
        # Auto-find latest results
        print(f"Auto-finding latest results in {args.results_dir}/...")

        baseline_files = sorted(glob.glob(str(args.results_dir / 'baseline_results_*.json')))
        mece_files = sorted(glob.glob(str(args.results_dir / 'mece_results_*.json')))

        if not baseline_files or not mece_files:
            print("❌ Could not find result files")
            return 1

        baseline_file = Path(baseline_files[-1])
        mece_file = Path(mece_files[-1])

        print(f"  Baseline: {baseline_file.name}")
        print(f"  MECE:     {mece_file.name}")

        # Load and analyze on the fly
        baseline_results, mece_results = load_results_files(baseline_file, mece_file)

        # Run analysis
        print("Running analysis...")
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_results import (
            compute_aggregate_stats,
            compare_conditions,
            statistical_significance_test,
            compute_per_category_stats
        )

        baseline_stats = compute_aggregate_stats(baseline_results)
        mece_stats = compute_aggregate_stats(mece_results)
        comparison = compare_conditions(baseline_stats, mece_stats)
        significance = statistical_significance_test(baseline_results, mece_results)
        baseline_by_cat = compute_per_category_stats(baseline_results)
        mece_by_cat = compute_per_category_stats(mece_results)

        analysis = {
            'comparison': comparison,
            'significance': significance,
            'baseline_by_category': baseline_by_cat,
            'mece_by_category': mece_by_cat,
        }

    # Generate documentation
    print("Generating documentation...")
    markdown = generate_markdown(analysis, baseline_results, mece_results)

    # Write output
    with open(args.output, 'w') as f:
        f.write(markdown)

    print(f"\n✅ Documentation generated: {args.output}")
    print(f"   {len(markdown)} characters, {len(markdown.split(chr(10)))} lines\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
