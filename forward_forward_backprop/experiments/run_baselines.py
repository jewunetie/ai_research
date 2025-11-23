"""
Run all baseline experiments for a given dataset.

This script automates running all baseline configurations:
1. Pure Backpropagation
2. Pure Forward-Forward
3. Sequential Phased (main hybrid approach)
4. Detached Interface (novel approach)

Usage:
    python experiments/run_baselines.py --dataset mnist
    python experiments/run_baselines.py --dataset fashion_mnist
    python experiments/run_baselines.py --dataset cifar10
    python experiments/run_baselines.py --all  # Run all datasets
"""

import argparse
import sys
from pathlib import Path
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(config_path: Path) -> bool:
    """
    Run a single experiment.

    Args:
        config_path: Path to config file

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print('='*80)

    try:
        # Run as subprocess to isolate each experiment
        result = subprocess.run(
            [sys.executable, 'experiments/run_experiment.py', '--config', str(config_path)],
            check=True,
            capture_output=False
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {config_path.name}")
        print(f"Error code: {e.returncode}")
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {config_path.name}")
        print(f"Error: {e}")
        return False


def run_baselines_for_dataset(dataset: str):
    """
    Run all baseline experiments for a specific dataset.

    Args:
        dataset: Dataset name (mnist, fashion_mnist, cifar10)
    """
    print(f"\n{'#'*80}")
    print(f"# RUNNING ALL BASELINES FOR: {dataset.upper()}")
    print(f"{'#'*80}\n")

    # Define configs to run for this dataset
    configs_dir = Path('forward_forward_backprop/experiments/configs')

    if dataset == 'mnist':
        configs = [
            'mnist_baseline_bp.yaml',
            'mnist_baseline_ff.yaml',
            'mnist_sequential_phased.yaml',
            'mnist_detached_interface.yaml'
        ]
    elif dataset == 'fashion_mnist':
        configs = [
            'fashion_mnist_sequential_phased.yaml',
            # Note: Would need to create fashion_mnist_baseline_bp.yaml, etc.
        ]
    elif dataset == 'cifar10':
        configs = [
            'cifar10_sequential_phased.yaml',
            # Note: Would need to create cifar10_baseline_bp.yaml, etc.
        ]
    else:
        print(f"Unknown dataset: {dataset}")
        return

    # Track results
    results = {}
    start_time = time.time()

    # Run each config
    for config_name in configs:
        config_path = configs_dir / config_name

        if not config_path.exists():
            print(f"⚠️  Config not found: {config_name} - skipping")
            results[config_name] = 'SKIPPED (not found)'
            continue

        success = run_experiment(config_path)
        results[config_name] = 'SUCCESS ✅' if success else 'FAILED ❌'

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BASELINE RUN SUMMARY - {dataset.upper()}")
    print('='*80)

    for config_name, status in results.items():
        print(f"  {config_name:<50} {status}")

    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run all baseline experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all MNIST baselines
  python experiments/run_baselines.py --dataset mnist

  # Run all Fashion-MNIST baselines
  python experiments/run_baselines.py --dataset fashion_mnist

  # Run all datasets
  python experiments/run_baselines.py --all
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'fashion_mnist', 'cifar10'],
        help='Dataset to run baselines for'
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Run baselines for all datasets'
    )

    args = parser.parse_args()

    if args.all:
        # Run all datasets
        for dataset in ['mnist', 'fashion_mnist', 'cifar10']:
            run_baselines_for_dataset(dataset)
    else:
        # Run single dataset
        run_baselines_for_dataset(args.dataset)


if __name__ == '__main__':
    main()
