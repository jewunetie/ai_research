"""
Compare results across multiple experiments.

Usage:
    python experiments/compare_results.py results/mnist_*

This script:
1. Loads results from multiple experiment directories
2. Compares final and best accuracies
3. Analyzes training curves
4. Generates comparison tables and plots
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import re

import torch


def parse_results_file(results_file: Path) -> Dict:
    """
    Parse a results text file into a dictionary.

    Args:
        results_file: Path to results_*.txt file

    Returns:
        Dictionary with experiment results
    """
    results = {}

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                results[key] = value

    return results


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file

    Returns:
        Checkpoint dictionary
    """
    return torch.load(checkpoint_path, map_location='cpu')


def find_experiments(result_dirs: List[str]) -> List[Dict]:
    """
    Find and load all experiments from result directories.

    Args:
        result_dirs: List of result directory patterns

    Returns:
        List of experiment dictionaries with metadata and results
    """
    experiments = []

    for pattern in result_dirs:
        # Expand glob pattern
        paths = list(Path('.').glob(pattern))

        for path in paths:
            if not path.is_dir():
                continue

            # Find most recent results file
            results_files = sorted(path.glob('results_*.txt'))
            if not results_files:
                print(f"Warning: No results file found in {path}")
                continue

            results_file = results_files[-1]  # Most recent
            results = parse_results_file(results_file)

            # Find checkpoints
            checkpoints = sorted(path.glob('checkpoint_*.pt'))

            experiments.append({
                'name': path.name,
                'path': path,
                'results': results,
                'results_file': results_file,
                'num_checkpoints': len(checkpoints),
                'has_final_model': (path / 'final_model.pt').exists()
            })

    return experiments


def compare_experiments(experiments: List[Dict]):
    """
    Print comparison table of experiments.

    Args:
        experiments: List of experiment dictionaries
    """
    if not experiments:
        print("No experiments found!")
        return

    print("=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)

    # Sort by final test accuracy (descending)
    experiments_sorted = sorted(
        experiments,
        key=lambda x: x['results'].get('final_test_acc', 0),
        reverse=True
    )

    # Print header
    print(f"\n{'Experiment':<40} {'Approach':<25} {'Final Acc':>10} {'Best Acc':>10} {'Best Epoch':>10}")
    print("-" * 100)

    # Print each experiment
    for exp in experiments_sorted:
        name = exp['name'][:39]  # Truncate if too long
        approach = str(exp['results'].get('Approach', 'Unknown'))[:24]
        final_acc = exp['results'].get('final_test_acc', 'N/A')
        best_acc = exp['results'].get('best_test_acc', 'N/A')
        best_epoch = exp['results'].get('best_epoch', 'N/A')

        # Format numbers
        if isinstance(final_acc, (int, float)):
            final_acc = f"{final_acc:.2f}%"
        if isinstance(best_acc, (int, float)):
            best_acc = f"{best_acc:.2f}%"

        print(f"{name:<40} {approach:<25} {str(final_acc):>10} {str(best_acc):>10} {str(best_epoch):>10}")

    print("-" * 100)

    # Summary statistics
    print("\nSUMMARY:")
    final_accs = [exp['results'].get('final_test_acc') for exp in experiments
                  if 'final_test_acc' in exp['results']]

    if final_accs:
        print(f"  Best final accuracy: {max(final_accs):.2f}%")
        print(f"  Worst final accuracy: {min(final_accs):.2f}%")
        print(f"  Average final accuracy: {sum(final_accs)/len(final_accs):.2f}%")
        print(f"  Std dev: {(sum((x - sum(final_accs)/len(final_accs))**2 for x in final_accs) / len(final_accs))**0.5:.2f}%")

    # Group by approach
    print("\nBY APPROACH:")
    approaches = {}
    for exp in experiments:
        approach = exp['results'].get('Approach', 'Unknown')
        if approach not in approaches:
            approaches[approach] = []
        final_acc = exp['results'].get('final_test_acc')
        if final_acc is not None:
            approaches[approach].append(final_acc)

    for approach, accs in sorted(approaches.items()):
        if accs:
            avg_acc = sum(accs) / len(accs)
            print(f"  {approach}: {avg_acc:.2f}% avg ({len(accs)} experiments)")

    # Group by dataset
    print("\nBY DATASET:")
    datasets = {}
    for exp in experiments:
        dataset = exp['results'].get('Dataset', 'Unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        final_acc = exp['results'].get('final_test_acc')
        if final_acc is not None:
            datasets[dataset].append(final_acc)

    for dataset, accs in sorted(datasets.items()):
        if accs:
            avg_acc = sum(accs) / len(accs)
            print(f"  {dataset}: {avg_acc:.2f}% avg ({len(accs)} experiments)")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Compare results across multiple experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all MNIST experiments
  python experiments/compare_results.py results/mnist_*

  # Compare specific experiments
  python experiments/compare_results.py results/mnist_baseline_bp results/mnist_sequential_phased

  # Compare all experiments
  python experiments/compare_results.py results/*
        """
    )

    parser.add_argument(
        'result_dirs',
        nargs='+',
        help='Result directory paths or patterns (supports wildcards)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed per-experiment information'
    )

    args = parser.parse_args()

    # Find and load experiments
    print("Loading experiments...")
    experiments = find_experiments(args.result_dirs)

    if not experiments:
        print("No experiments found matching the patterns!")
        print(f"Searched: {args.result_dirs}")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments\n")

    # Show comparison
    compare_experiments(experiments)

    # Detailed view if requested
    if args.detailed:
        print("\n" + "=" * 100)
        print("DETAILED EXPERIMENT INFO")
        print("=" * 100)

        for exp in experiments:
            print(f"\n{exp['name']}")
            print("-" * 60)
            print(f"  Path: {exp['path']}")
            print(f"  Results file: {exp['results_file']}")
            print(f"  Checkpoints: {exp['num_checkpoints']}")
            print(f"  Final model: {'Yes' if exp['has_final_model'] else 'No'}")
            print(f"  Results:")
            for key, value in exp['results'].items():
                print(f"    {key}: {value}")


if __name__ == '__main__':
    main()
