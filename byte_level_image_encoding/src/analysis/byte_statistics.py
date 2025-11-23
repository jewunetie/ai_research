#!/usr/bin/env python3
"""
Byte pattern statistics analyzer.

This module analyzes byte-level patterns across different image formats
to understand format-specific signatures and differences.

Usage:
    python src/analysis/byte_statistics.py --data-dir PATH

Example:
    python src/analysis/byte_statistics.py \
        --data-dir data/cifar10 \
        --num-samples 1000
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def analyze_byte_distribution(
    data_dir: Path,
    format_name: str,
    num_samples: int = 1000
) -> Dict:
    """
    Analyze byte value distribution for a format.

    Args:
        data_dir: Directory containing image files
        format_name: Format name (jpeg_q75, png, etc.)
        num_samples: Number of samples to analyze

    Returns:
        Dict with distribution statistics
    """
    format_dir = data_dir / format_name / 'test'

    if not format_dir.exists():
        raise FileNotFoundError(f"Directory not found: {format_dir}")

    files = sorted(list(format_dir.glob('*.*')))[:num_samples]

    # Byte frequency (0-255)
    byte_freq = np.zeros(256, dtype=np.int64)

    # File sizes
    file_sizes = []

    # Byte sequences (for entropy calculation)
    all_bytes = []

    # First/last byte patterns
    first_bytes = []
    last_bytes = []

    print(f"  Analyzing {len(files)} files from {format_name}...")

    for file_path in tqdm(files, desc=f"  {format_name}", leave=False):
        with open(file_path, 'rb') as f:
            data = f.read()

        file_sizes.append(len(data))
        all_bytes.extend(data)

        # Update frequency
        for byte_val in data:
            byte_freq[byte_val] += 1

        # First/last bytes
        if len(data) > 0:
            first_bytes.append(data[0])
        if len(data) > 10:
            last_bytes.extend(data[-10:])

    # Calculate statistics
    all_bytes = np.array(all_bytes, dtype=np.uint8)

    # Entropy
    byte_prob = byte_freq / byte_freq.sum()
    byte_prob = byte_prob[byte_prob > 0]
    entropy = -np.sum(byte_prob * np.log2(byte_prob))

    # Most common bytes
    most_common_indices = np.argsort(byte_freq)[::-1][:20]
    most_common = [(int(idx), int(byte_freq[idx])) for idx in most_common_indices]

    return {
        'format': format_name,
        'num_files': len(files),
        'byte_frequency': byte_freq,
        'file_sizes': file_sizes,
        'avg_file_size': np.mean(file_sizes),
        'std_file_size': np.std(file_sizes),
        'min_file_size': np.min(file_sizes),
        'max_file_size': np.max(file_sizes),
        'entropy': entropy,
        'most_common_bytes': most_common,
        'first_bytes': np.array(first_bytes),
        'last_bytes': np.array(last_bytes),
        'unique_bytes': np.sum(byte_freq > 0),
    }


def plot_byte_distributions(stats_list: List[Dict], output_path: Path):
    """
    Plot byte frequency distributions for all formats.

    Args:
        stats_list: List of statistics dicts
        output_path: Path to save figure
    """
    n_formats = len(stats_list)

    fig, axes = plt.subplots(n_formats, 1, figsize=(14, 4*n_formats))
    if n_formats == 1:
        axes = [axes]

    for ax, stats in zip(axes, stats_list):
        fmt = stats['format']
        byte_freq = stats['byte_frequency']

        # Normalize to probability
        byte_prob = byte_freq / byte_freq.sum()

        x = np.arange(256)
        ax.bar(x, byte_prob, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Byte Value (0-255)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(f'{fmt.upper()} - Byte Distribution (Entropy: {stats["entropy"]:.2f} bits)',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 255)

        # Highlight magic bytes (file headers)
        if len(stats['first_bytes']) > 0:
            unique_first = np.unique(stats['first_bytes'])
            for byte_val in unique_first[:5]:  # Top 5 magic bytes
                ax.axvline(x=byte_val, color='red', linestyle='--',
                          alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_file_size_comparison(stats_list: List[Dict], output_path: Path):
    """
    Compare file sizes across formats.

    Args:
        stats_list: List of statistics dicts
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    formats = [s['format'].upper() for s in stats_list]
    avg_sizes = [s['avg_file_size'] for s in stats_list]
    std_sizes = [s['std_file_size'] for s in stats_list]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(formats)]

    bars = ax.bar(formats, avg_sizes, yerr=std_sizes, color=colors,
                  alpha=0.8, edgecolor='black', capsize=10)

    # Add value labels
    for bar, avg, std in zip(bars, avg_sizes, std_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.0f} ±{std:.0f}\nbytes',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('File Size (bytes)', fontsize=12, fontweight='bold')
    ax.set_title('Average File Size by Format (CIFAR-10 32×32)',
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_entropy_comparison(stats_list: List[Dict], output_path: Path):
    """
    Compare byte entropy across formats.

    Args:
        stats_list: List of statistics dicts
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    formats = [s['format'].upper() for s in stats_list]
    entropies = [s['entropy'] for s in stats_list]

    # Color by entropy (higher = more compressed/complex)
    colors = plt.cm.viridis(np.array(entropies) / 8.0)  # Max entropy = 8 bits

    bars = ax.bar(formats, entropies, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, ent in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ent:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Byte Entropy by Format (Higher = More Random/Compressed)',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_ylim(0, 8.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add reference line for maximum entropy
    ax.axhline(y=8.0, color='red', linestyle='--', linewidth=2,
              label='Maximum Entropy (8 bits)', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_magic_bytes(stats_list: List[Dict], output_path: Path):
    """
    Visualize magic bytes (file headers) for each format.

    Args:
        stats_list: List of statistics dicts
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(len(stats_list), 1, figsize=(12, 3*len(stats_list)))
    if len(stats_list) == 1:
        axes = [axes]

    for ax, stats in zip(axes, stats_list):
        fmt = stats['format']
        first_bytes = stats['first_bytes']

        # Count frequency of first bytes
        unique, counts = np.unique(first_bytes, return_counts=True)

        # Take top 10
        top_indices = np.argsort(counts)[::-1][:10]
        top_bytes = unique[top_indices]
        top_counts = counts[top_indices]

        # Convert to hex for display
        hex_labels = [f'0x{b:02X}\n({b})' for b in top_bytes]

        bars = ax.bar(range(len(top_bytes)), top_counts, color='steelblue',
                     alpha=0.8, edgecolor='black')

        ax.set_xticks(range(len(top_bytes)))
        ax.set_xticklabels(hex_labels, fontsize=9)
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{fmt.upper()} - First Byte (Magic Byte) Distribution',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def generate_statistics_report(stats_list: List[Dict], output_path: Path):
    """
    Generate text report with statistics.

    Args:
        stats_list: List of statistics dicts
        output_path: Path to save report
    """
    lines = []

    lines.append("=" * 80)
    lines.append("BYTE PATTERN STATISTICS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for stats in stats_list:
        fmt = stats['format']

        lines.append(f"## {fmt.upper()}")
        lines.append("")

        lines.append("### File Size Statistics")
        lines.append(f"  Average: {stats['avg_file_size']:.0f} bytes")
        lines.append(f"  Std Dev: {stats['std_file_size']:.0f} bytes")
        lines.append(f"  Min: {stats['min_file_size']} bytes")
        lines.append(f"  Max: {stats['max_file_size']} bytes")
        lines.append("")

        lines.append("### Byte Distribution")
        lines.append(f"  Entropy: {stats['entropy']:.2f} bits (max: 8.0)")
        lines.append(f"  Unique byte values used: {stats['unique_bytes']}/256")
        lines.append("")

        lines.append("### Most Common Bytes")
        for i, (byte_val, count) in enumerate(stats['most_common_bytes'][:10], 1):
            prob = count / stats['byte_frequency'].sum()
            lines.append(f"  {i}. Byte 0x{byte_val:02X} ({byte_val:3d}): "
                        f"{count:>8,} occurrences ({prob*100:.2f}%)")
        lines.append("")

        # Magic bytes
        if len(stats['first_bytes']) > 0:
            unique_first, counts_first = np.unique(stats['first_bytes'], return_counts=True)
            lines.append("### Magic Bytes (First Byte of File)")
            for byte_val, count in zip(unique_first[:5], counts_first[:5]):
                lines.append(f"  0x{byte_val:02X} ({byte_val:3d}): {count} files")
            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    # Comparative analysis
    lines.append("## COMPARATIVE ANALYSIS")
    lines.append("")

    # Sort by entropy
    sorted_by_entropy = sorted(stats_list, key=lambda x: x['entropy'], reverse=True)
    lines.append("### Format Complexity (by Entropy)")
    for i, stats in enumerate(sorted_by_entropy, 1):
        lines.append(f"  {i}. {stats['format'].upper()}: {stats['entropy']:.2f} bits")
    lines.append("")

    lines.append("**Interpretation**:")
    highest_entropy = sorted_by_entropy[0]
    lowest_entropy = sorted_by_entropy[-1]
    lines.append(f"  - {highest_entropy['format'].upper()} has highest entropy "
                f"({highest_entropy['entropy']:.2f} bits)")
    lines.append(f"    → Most compressed/random byte patterns")
    lines.append(f"  - {lowest_entropy['format'].upper()} has lowest entropy "
                f"({lowest_entropy['entropy']:.2f} bits)")
    lines.append(f"    → More structured/predictable byte patterns")
    lines.append("")

    # Sort by file size
    sorted_by_size = sorted(stats_list, key=lambda x: x['avg_file_size'])
    lines.append("### File Size Efficiency")
    for i, stats in enumerate(sorted_by_size, 1):
        lines.append(f"  {i}. {stats['format'].upper()}: {stats['avg_file_size']:.0f} bytes")
    lines.append("")

    lines.append("=" * 80)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Saved: {output_path}")


def main(args):
    """Main byte statistics analysis function."""

    print("=" * 70)
    print("Byte Pattern Statistics Analysis")
    print("=" * 70)

    data_dir = Path(args.data_dir)

    # Formats to analyze
    formats = ['jpeg_q75', 'png', 'webp', 'bmp']
    available_formats = [fmt for fmt in formats if (data_dir / fmt / 'test').exists()]

    if not available_formats:
        print(f"\nError: No format directories found in {data_dir}")
        print(f"Expected: {formats}")
        return

    print(f"\nFound formats: {', '.join(available_formats)}")
    print(f"Analyzing {args.num_samples} samples per format...")

    # Analyze each format
    stats_list = []
    for fmt in available_formats:
        print(f"\n{fmt.upper()}:")
        stats = analyze_byte_distribution(data_dir, fmt, args.num_samples)
        stats_list.append(stats)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_byte_distributions(stats_list, output_dir / 'byte_distributions.png')
    plot_file_size_comparison(stats_list, output_dir / 'file_size_comparison.png')
    plot_entropy_comparison(stats_list, output_dir / 'entropy_comparison.png')
    plot_magic_bytes(stats_list, output_dir / 'magic_bytes.png')

    # Generate report
    print("\nGenerating text report...")
    generate_statistics_report(stats_list, output_dir / 'byte_statistics_report.txt')

    # Save raw data
    import json

    json_data = []
    for stats in stats_list:
        json_stats = {
            'format': stats['format'],
            'num_files': stats['num_files'],
            'avg_file_size': float(stats['avg_file_size']),
            'std_file_size': float(stats['std_file_size']),
            'min_file_size': int(stats['min_file_size']),
            'max_file_size': int(stats['max_file_size']),
            'entropy': float(stats['entropy']),
            'unique_bytes': int(stats['unique_bytes']),
            'most_common_bytes': stats['most_common_bytes'][:20],
        }
        json_data.append(json_stats)

    with open(output_dir / 'byte_statistics_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  ✓ Saved: {output_dir / 'byte_statistics_data.json'}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - byte_distributions.png")
    print("  - file_size_comparison.png")
    print("  - entropy_comparison.png")
    print("  - magic_bytes.png")
    print("  - byte_statistics_report.txt")
    print("  - byte_statistics_data.json")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze byte patterns across image formats'
    )

    parser.add_argument('--data-dir', type=str, default='./data/cifar10',
                        help='Base data directory containing format subdirectories')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to analyze per format')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/h1_zero_shot_transfer/results/byte_statistics',
                        help='Output directory for analysis')

    args = parser.parse_args()

    main(args)
