#!/usr/bin/env python3
"""
Run MECE evaluation on math case analysis dataset.

This script evaluates both baseline and MECE conditions on the dataset.
Can run with or without actual model (uses mock responses for testing).

Usage:
    # Test with mock responses (no model required)
    python scripts/run_evaluation.py --test --limit 5

    # Full evaluation with model (requires dependencies)
    python scripts/run_evaluation.py --condition baseline --limit 50
    python scripts/run_evaluation.py --condition mece --mece-version 1 --limit 50

    # Both conditions
    python scripts/run_evaluation.py --both --limit 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import MECEEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run MECE evaluation")

    parser.add_argument(
        '--condition',
        type=str,
        choices=['baseline', 'mece', 'both'],
        default='both',
        help='Condition to evaluate (default: both)'
    )

    parser.add_argument(
        '--mece-version',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='MECE prompt version to use (default: 1)'
    )

    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/math_case_analysis.json'),
        help='Path to dataset JSON file'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems to evaluate (default: all)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: use mock responses (no model required)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed evaluation information'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Output directory for results (default: results/)'
    )

    args = parser.parse_args()

    # Check dataset exists
    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return 1

    # Initialize model inference (if not in test mode)
    model_inference = None
    if not args.test:
        try:
            from src.models.qwen_inference import QwenInference
            print("Loading Qwen3-0.6B-Instruct model...")
            model_inference = QwenInference()
            print("✅ Model loaded successfully")
        except ImportError as e:
            print(f"⚠️  Could not load model: {e}")
            print("   Running in test mode with mock responses")
            model_inference = None

    # Create evaluator
    evaluator = MECEEvaluator(
        model_inference=model_inference,
        mece_version=args.mece_version,
        verbose=args.verbose,
        output_dir=args.output_dir
    )

    # Run evaluation
    conditions = []
    if args.condition == 'both':
        conditions = ['baseline', 'mece']
    else:
        conditions = [args.condition]

    all_results = {}
    for condition in conditions:
        print(f"\n{'='*70}")
        print(f"Running {condition.upper()} evaluation...")
        print(f"{'='*70}\n")

        results = evaluator.evaluate_dataset(
            dataset_path=args.dataset,
            condition=condition,
            limit=args.limit
        )

        all_results[condition] = results

        # Print summary
        print(f"\n{'='*70}")
        print(f"{condition.upper()} RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Problems evaluated: {len(results)}")

        # Aggregate metrics
        accuracies = [r['metrics']['accuracy']['f1_score'] for r in results]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"Average Accuracy F1: {avg_accuracy:.3f}")

        if 'mutual_exclusivity' in results[0]['metrics']:
            me_scores = [
                r['metrics']['mutual_exclusivity'].get('overall_me_score', 0)
                for r in results
                if 'error' not in r['metrics']['mutual_exclusivity']
            ]
            if me_scores:
                avg_me = sum(me_scores) / len(me_scores)
                print(f"Average ME Score: {avg_me:.3f}")

        if 'collective_exhaustiveness' in results[0]['metrics']:
            ce_scores = [
                r['metrics']['collective_exhaustiveness'].get('overall_ce_score', 0)
                for r in results
                if 'error' not in r['metrics']['collective_exhaustiveness']
            ]
            if ce_scores:
                avg_ce = sum(ce_scores) / len(ce_scores)
                print(f"Average CE Score: {avg_ce:.3f}")

        avg_latency = sum(r['latency_ms'] for r in results) / len(results)
        print(f"Average Latency: {avg_latency:.0f}ms")

    # Comparison if both conditions run
    if len(conditions) == 2:
        print(f"\n{'='*70}")
        print("BASELINE vs MECE COMPARISON")
        print(f"{'='*70}")

        baseline_results = all_results['baseline']
        mece_results = all_results['mece']

        baseline_acc = sum(r['metrics']['accuracy']['f1_score'] for r in baseline_results) / len(baseline_results)
        mece_acc = sum(r['metrics']['accuracy']['f1_score'] for r in mece_results) / len(mece_results)

        print(f"Accuracy F1:")
        print(f"  Baseline: {baseline_acc:.3f}")
        print(f"  MECE:     {mece_acc:.3f}")
        print(f"  Δ:        {mece_acc - baseline_acc:+.3f}")

    print(f"\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
