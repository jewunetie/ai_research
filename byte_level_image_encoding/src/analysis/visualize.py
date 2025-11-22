#!/usr/bin/env python3
"""
Visualization script for zero-shot transfer results.

Creates publication-quality plots for the format-agnostic transfer experiment:
1. Accuracy by format (bar chart)
2. Transfer ratio plot
3. Per-class transfer heatmap

Usage:
    python src/analysis/visualize.py --results PATH [--output-dir PATH]

Example:
    python src/analysis/visualize.py \
        --results experiments/h1_zero_shot_transfer/results/zero_shot_results.json \
        --output-dir experiments/h1_zero_shot_transfer/results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_accuracy_by_format(results, trained_format, output_path):
    """
    Plot accuracy by format (bar chart).

    Args:
        results: Dict of format -> results
        trained_format: Name of training format
        output_path: Path to save figure
    """
    formats = list(results.keys())
    accuracies = [results[fmt]['accuracy'] for fmt in formats]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors: trained format in blue, zero-shot in orange
    colors = ['#3498db' if fmt == trained_format else '#e74c3c' for fmt in formats]

    # Bar chart
    bars = ax.bar(range(len(formats)), accuracies, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xlabel('Image Format', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('ByteFormer Accuracy Across Image Formats\n(Trained on JPEG only)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels([fmt.upper() for fmt in formats], fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='In-distribution (trained)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Zero-shot transfer')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_transfer_ratios(results, trained_format, output_path):
    """
    Plot transfer ratios relative to in-distribution accuracy.

    Args:
        results: Dict of format -> results
        trained_format: Name of training format
        output_path: Path to save figure
    """
    baseline_acc = results[trained_format]['accuracy']

    # Calculate transfer ratios
    formats = [fmt for fmt in results.keys() if fmt != trained_format]
    transfer_ratios = [
        (results[fmt]['accuracy'] / baseline_acc) * 100
        for fmt in formats
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color code by transfer quality
    colors = []
    for ratio in transfer_ratios:
        if ratio >= 70:
            colors.append('#27ae60')  # Green (strong)
        elif ratio >= 40:
            colors.append('#f39c12')  # Orange (partial)
        else:
            colors.append('#e74c3c')  # Red (weak)

    # Bar chart
    bars = ax.bar(range(len(formats)), transfer_ratios, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, ratio in zip(bars, transfer_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # H1 threshold line
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='H1 Threshold (50%)', alpha=0.7)

    # Formatting
    ax.set_xlabel('Zero-Shot Format', fontsize=13, fontweight='bold')
    ax.set_ylabel('Transfer Ratio (% of in-distribution accuracy)', fontsize=13, fontweight='bold')
    ax.set_title(f'Zero-Shot Transfer Performance\n(Baseline: {trained_format.upper()} = {baseline_acc:.1f}%)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels([fmt.upper() for fmt in formats], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Strong transfer (≥70%)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Partial transfer (40-70%)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Weak transfer (<40%)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='H1 Threshold (50%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_per_class_heatmap(results, output_path):
    """
    Plot per-class accuracy heatmap across formats.

    Args:
        results: Dict of format -> results
        output_path: Path to save figure
    """
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    formats = list(results.keys())

    # Build matrix: [classes x formats]
    matrix = np.zeros((10, len(formats)))
    for j, fmt in enumerate(formats):
        for i in range(10):
            matrix[i, j] = results[fmt]['class_accuracies'][str(i)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        xticklabels=[fmt.upper() for fmt in formats],
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Formatting
    ax.set_xlabel('Image Format', fontsize=13, fontweight='bold')
    ax.set_ylabel('CIFAR-10 Class', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Accuracy Across Formats',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_class_transfer_analysis(results, trained_format, output_path):
    """
    Analyze which classes transfer well vs poorly.

    Args:
        results: Dict of format -> results
        trained_format: Name of training format
        output_path: Path to save figure
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Calculate average transfer ratio per class
    zero_shot_formats = [fmt for fmt in results.keys() if fmt != trained_format]

    class_transfer_ratios = []
    for class_idx in range(10):
        baseline = results[trained_format]['class_accuracies'][str(class_idx)]
        if baseline > 0:
            avg_transfer = np.mean([
                results[fmt]['class_accuracies'][str(class_idx)] / baseline
                for fmt in zero_shot_formats
            ])
            class_transfer_ratios.append(avg_transfer * 100)
        else:
            class_transfer_ratios.append(0)

    # Sort by transfer ratio
    sorted_indices = np.argsort(class_transfer_ratios)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color code
    colors = ['#27ae60' if r >= 70 else '#f39c12' if r >= 40 else '#e74c3c'
              for r in class_transfer_ratios]
    colors_sorted = [colors[i] for i in sorted_indices]

    # Bar chart
    bars = ax.barh(
        range(10),
        [class_transfer_ratios[i] for i in sorted_indices],
        color=colors_sorted,
        alpha=0.8,
        edgecolor='black'
    )

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # H1 threshold line
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Formatting
    ax.set_xlabel('Average Transfer Ratio (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('CIFAR-10 Class', fontsize=13, fontweight='bold')
    ax.set_title('Which Classes Transfer Best?\n(Average across zero-shot formats)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(range(10))
    ax.set_yticklabels([class_names[i] for i in sorted_indices], fontsize=11)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def main(args):
    """Main visualization function."""

    print("=" * 70)
    print("Visualization: Zero-Shot Transfer Results")
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

    print(f"\nGenerating visualizations...")

    # 1. Accuracy by format
    plot_accuracy_by_format(
        results,
        trained_format,
        output_dir / 'accuracy_by_format.png'
    )

    # 2. Transfer ratios
    plot_transfer_ratios(
        results,
        trained_format,
        output_dir / 'transfer_ratios.png'
    )

    # 3. Per-class heatmap
    plot_per_class_heatmap(
        results,
        output_dir / 'per_class_heatmap.png'
    )

    # 4. Class transfer analysis
    plot_class_transfer_analysis(
        results,
        trained_format,
        output_dir / 'class_transfer_analysis.png'
    )

    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  1. accuracy_by_format.png - Overall accuracy comparison")
    print("  2. transfer_ratios.png - Transfer performance vs H1 threshold")
    print("  3. per_class_heatmap.png - Per-class accuracy matrix")
    print("  4. class_transfer_analysis.png - Which classes transfer best")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize zero-shot transfer results'
    )

    parser.add_argument('--results', type=str, required=True,
                        help='Path to zero_shot_results.json')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/h1_zero_shot_transfer/results/figures',
                        help='Output directory for figures')

    args = parser.parse_args()

    main(args)
