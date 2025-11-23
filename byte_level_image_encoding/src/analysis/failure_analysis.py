#!/usr/bin/env python3
"""
Failure analysis for zero-shot format transfer.

This module provides detailed analysis of transfer failures:
- Which classes transfer well vs poorly
- Simple vs complex class patterns
- Format-specific failure modes
- Confusion matrix analysis

Usage:
    python src/analysis/failure_analysis.py --results PATH

Example:
    python src/analysis/failure_analysis.py \
        --results experiments/h1_zero_shot_transfer/results/zero_shot_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Class complexity categories (based on intra-class variation)
SIMPLE_CLASSES = ['airplane', 'automobile', 'ship', 'truck']  # Vehicles with consistent shapes
COMPLEX_CLASSES = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']  # Animals with high variation


def analyze_class_transfer(results: Dict, trained_format: str) -> Dict:
    """
    Analyze transfer performance by class.

    Args:
        results: Dict of format -> results
        trained_format: Name of training format

    Returns:
        Dict with class transfer analysis
    """
    baseline_class_acc = results[trained_format]['class_accuracies']

    class_analysis = {}

    for class_idx in range(10):
        class_name = CLASS_NAMES[class_idx]
        baseline_acc = baseline_class_acc[str(class_idx)]

        # Calculate transfer for each format
        transfer_ratios = {}
        for fmt in results.keys():
            if fmt != trained_format:
                zero_shot_acc = results[fmt]['class_accuracies'][str(class_idx)]
                transfer_ratio = (zero_shot_acc / baseline_acc * 100) if baseline_acc > 0 else 0
                transfer_ratios[fmt] = transfer_ratio

        # Average transfer ratio
        avg_transfer = np.mean(list(transfer_ratios.values())) if transfer_ratios else 0

        class_analysis[class_name] = {
            'class_idx': class_idx,
            'baseline_accuracy': baseline_acc,
            'transfer_ratios': transfer_ratios,
            'avg_transfer_ratio': avg_transfer,
            'is_simple': class_name in SIMPLE_CLASSES,
        }

    return class_analysis


def analyze_simple_vs_complex(class_analysis: Dict) -> Dict:
    """
    Compare transfer performance between simple and complex classes.

    Args:
        class_analysis: Dict from analyze_class_transfer()

    Returns:
        Dict with simple vs complex comparison
    """
    simple_transfers = []
    complex_transfers = []

    for class_name, data in class_analysis.items():
        if data['is_simple']:
            simple_transfers.append(data['avg_transfer_ratio'])
        else:
            complex_transfers.append(data['avg_transfer_ratio'])

    return {
        'simple_classes': {
            'classes': SIMPLE_CLASSES,
            'avg_transfer': np.mean(simple_transfers),
            'std_transfer': np.std(simple_transfers),
            'transfers': simple_transfers,
        },
        'complex_classes': {
            'classes': COMPLEX_CLASSES,
            'avg_transfer': np.mean(complex_transfers),
            'std_transfer': np.std(complex_transfers),
            'transfers': complex_transfers,
        },
        'difference': np.mean(simple_transfers) - np.mean(complex_transfers),
    }


def analyze_format_patterns(results: Dict, trained_format: str) -> Dict:
    """
    Analyze format-specific patterns.

    Args:
        results: Dict of format -> results
        trained_format: Name of training format

    Returns:
        Dict with format pattern analysis
    """
    baseline_acc = results[trained_format]['accuracy']

    format_analysis = {}

    for fmt in results.keys():
        if fmt == trained_format:
            continue

        fmt_acc = results[fmt]['accuracy']
        transfer_ratio = (fmt_acc / baseline_acc * 100) if baseline_acc > 0 else 0

        # Identify which classes fail most
        class_drops = []
        for class_idx in range(10):
            baseline_class = results[trained_format]['class_accuracies'][str(class_idx)]
            fmt_class = results[fmt]['class_accuracies'][str(class_idx)]
            drop = baseline_class - fmt_class
            class_drops.append({
                'class': CLASS_NAMES[class_idx],
                'drop': drop,
                'baseline': baseline_class,
                'zero_shot': fmt_class,
            })

        # Sort by drop
        class_drops.sort(key=lambda x: x['drop'], reverse=True)

        format_analysis[fmt] = {
            'accuracy': fmt_acc,
            'transfer_ratio': transfer_ratio,
            'top_failures': class_drops[:5],  # Top 5 failures
            'top_successes': sorted(class_drops, key=lambda x: x['drop'])[:5],  # Top 5 successes
        }

    return format_analysis


def plot_simple_vs_complex(simple_vs_complex: Dict, output_path: Path):
    """
    Plot simple vs complex class transfer comparison.

    Args:
        simple_vs_complex: Dict from analyze_simple_vs_complex()
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Simple Classes\n(Vehicles)', 'Complex Classes\n(Animals)']
    means = [
        simple_vs_complex['simple_classes']['avg_transfer'],
        simple_vs_complex['complex_classes']['avg_transfer']
    ]
    stds = [
        simple_vs_complex['simple_classes']['std_transfer'],
        simple_vs_complex['complex_classes']['std_transfer']
    ]

    colors = ['#27ae60', '#e74c3c']
    bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8,
                   edgecolor='black', capsize=10)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # H1 threshold line
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2,
               label='H1 Threshold (50%)', alpha=0.7)

    # Formatting
    ax.set_ylabel('Average Transfer Ratio (%)', fontsize=13, fontweight='bold')
    ax.set_title('Simple vs Complex Classes: Transfer Performance',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Add explanation
    diff = simple_vs_complex['difference']
    explanation = f"Difference: {diff:+.1f}% ({'Simple better' if diff > 0 else 'Complex better'})"
    ax.text(0.5, 0.95, explanation, transform=ax.transAxes,
            ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_failure_modes(format_analysis: Dict, output_path: Path):
    """
    Plot format-specific failure modes.

    Args:
        format_analysis: Dict from analyze_format_patterns()
        output_path: Path to save figure
    """
    formats = list(format_analysis.keys())
    n_formats = len(formats)

    fig, axes = plt.subplots(1, n_formats, figsize=(6*n_formats, 6))
    if n_formats == 1:
        axes = [axes]

    for ax, fmt in zip(axes, formats):
        failures = format_analysis[fmt]['top_failures']

        classes = [f['class'] for f in failures]
        drops = [f['drop'] for f in failures]

        colors = ['#e74c3c' if drop > 20 else '#f39c12' if drop > 10 else '#95a5a6'
                  for drop in drops]

        bars = ax.barh(range(len(classes)), drops, color=colors,
                       alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, drop) in enumerate(zip(bars, drops)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {drop:.1f}%',
                    ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel('Accuracy Drop (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{fmt.upper()}\nTop 5 Failure Classes',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def generate_text_report(
    class_analysis: Dict,
    simple_vs_complex: Dict,
    format_analysis: Dict,
    output_path: Path
):
    """
    Generate text report with detailed findings.

    Args:
        class_analysis: Dict from analyze_class_transfer()
        simple_vs_complex: Dict from analyze_simple_vs_complex()
        format_analysis: Dict from analyze_format_patterns()
        output_path: Path to save report
    """
    lines = []

    lines.append("=" * 80)
    lines.append("FAILURE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("## EXECUTIVE SUMMARY")
    lines.append("")

    # Best/worst classes
    sorted_classes = sorted(class_analysis.items(),
                           key=lambda x: x[1]['avg_transfer_ratio'], reverse=True)

    lines.append("### Best Transferring Classes")
    for i, (class_name, data) in enumerate(sorted_classes[:3], 1):
        lines.append(f"{i}. {class_name.upper()}: {data['avg_transfer_ratio']:.1f}% "
                    f"(baseline: {data['baseline_accuracy']:.1f}%)")
    lines.append("")

    lines.append("### Worst Transferring Classes")
    for i, (class_name, data) in enumerate(sorted_classes[-3:], 1):
        lines.append(f"{i}. {class_name.upper()}: {data['avg_transfer_ratio']:.1f}% "
                    f"(baseline: {data['baseline_accuracy']:.1f}%)")
    lines.append("")

    # Simple vs Complex
    lines.append("## SIMPLE VS COMPLEX CLASSES")
    lines.append("")

    simple = simple_vs_complex['simple_classes']
    complex = simple_vs_complex['complex_classes']

    lines.append(f"Simple classes (vehicles): {simple['avg_transfer']:.1f}% ± {simple['std_transfer']:.1f}%")
    lines.append(f"  Classes: {', '.join(simple['classes'])}")
    lines.append("")

    lines.append(f"Complex classes (animals): {complex['avg_transfer']:.1f}% ± {complex['std_transfer']:.1f}%")
    lines.append(f"  Classes: {', '.join(complex['classes'])}")
    lines.append("")

    diff = simple_vs_complex['difference']
    if abs(diff) > 5:
        if diff > 0:
            lines.append(f"**FINDING**: Simple classes transfer {diff:.1f}% better than complex classes")
            lines.append("  → Consistent shapes (vehicles) are more format-agnostic")
        else:
            lines.append(f"**FINDING**: Complex classes transfer {-diff:.1f}% better than simple classes")
            lines.append("  → Surprising! Animals are more format-agnostic than vehicles")
    else:
        lines.append(f"**FINDING**: No significant difference ({diff:.1f}%) between simple and complex")
        lines.append("  → Class complexity doesn't affect format transfer")
    lines.append("")

    # Format-specific patterns
    lines.append("## FORMAT-SPECIFIC FAILURE MODES")
    lines.append("")

    for fmt, data in format_analysis.items():
        lines.append(f"### {fmt.upper()}")
        lines.append(f"Transfer ratio: {data['transfer_ratio']:.1f}%")
        lines.append("")

        lines.append("Top 3 Failures:")
        for i, failure in enumerate(data['top_failures'][:3], 1):
            lines.append(f"  {i}. {failure['class']}: "
                        f"{failure['baseline']:.1f}% → {failure['zero_shot']:.1f}% "
                        f"(drop: {failure['drop']:.1f}%)")
        lines.append("")

        lines.append("Top 3 Successes:")
        for i, success in enumerate(data['top_successes'][:3], 1):
            lines.append(f"  {i}. {success['class']}: "
                        f"{success['baseline']:.1f}% → {success['zero_shot']:.1f}% "
                        f"(drop: {success['drop']:.1f}%)")
        lines.append("")

    # Recommendations
    lines.append("## RECOMMENDATIONS")
    lines.append("")

    # Identify consistent failures
    consistent_failures = set()
    for fmt, data in format_analysis.items():
        for failure in data['top_failures'][:2]:
            consistent_failures.add(failure['class'])

    if len(consistent_failures) >= 2:
        lines.append("### Classes that consistently fail across formats:")
        for cls in sorted(consistent_failures):
            lines.append(f"  - {cls}")
        lines.append("")
        lines.append("**Recommendation**: Focus analysis on these classes")
        lines.append("  → What makes them format-dependent?")
        lines.append("  → Visualize attention patterns")
        lines.append("  → Examine byte-level differences")
        lines.append("")

    lines.append("=" * 80)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Saved: {output_path}")


def main(args):
    """Main failure analysis function."""

    print("=" * 70)
    print("Failure Analysis")
    print("=" * 70)

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"\nLoading results: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    trained_format = data['trained_format']

    print(f"Trained format: {trained_format}")
    print(f"Formats evaluated: {', '.join(results.keys())}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning analysis...")

    # 1. Class transfer analysis
    print("  Analyzing class-level transfer...")
    class_analysis = analyze_class_transfer(results, trained_format)

    # 2. Simple vs complex
    print("  Comparing simple vs complex classes...")
    simple_vs_complex = analyze_simple_vs_complex(class_analysis)

    # 3. Format patterns
    print("  Analyzing format-specific patterns...")
    format_analysis = analyze_format_patterns(results, trained_format)

    # Generate plots
    print("\nGenerating visualizations...")

    plot_simple_vs_complex(
        simple_vs_complex,
        output_dir / 'simple_vs_complex.png'
    )

    plot_failure_modes(
        format_analysis,
        output_dir / 'failure_modes.png'
    )

    # Generate text report
    print("\nGenerating text report...")
    generate_text_report(
        class_analysis,
        simple_vs_complex,
        format_analysis,
        output_dir / 'failure_analysis_report.txt'
    )

    # Save analysis data
    analysis_data = {
        'class_analysis': class_analysis,
        'simple_vs_complex': simple_vs_complex,
        'format_analysis': format_analysis,
    }

    with open(output_dir / 'failure_analysis_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"  ✓ Saved: {output_dir / 'failure_analysis_data.json'}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - simple_vs_complex.png")
    print("  - failure_modes.png")
    print("  - failure_analysis_report.txt")
    print("  - failure_analysis_data.json")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze zero-shot transfer failures'
    )

    parser.add_argument('--results', type=str, required=True,
                        help='Path to zero_shot_results.json')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/h1_zero_shot_transfer/results/failure_analysis',
                        help='Output directory for analysis')

    args = parser.parse_args()

    main(args)
