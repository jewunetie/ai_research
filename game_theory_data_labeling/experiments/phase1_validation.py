"""Phase 1 validation experiments.

Experiment 1a: All truthful agents (high ability) → should achieve >90% accuracy
Experiment 1b: All lazy agents (random) → should achieve ≈50% accuracy
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine


def run_experiment_1a():
    """Experiment 1a: High-ability truthful agents.

    Expected: >90% accuracy with both Majority Voting and Dawid-Skene.
    """
    print("=" * 70)
    print("EXPERIMENT 1a: High-ability truthful agents")
    print("=" * 70)
    print()

    # Configuration
    config_mv = ExperimentConfig(
        name="exp1a_majority_voting",
        num_tasks=50,
        num_agents=10,
        agent_mix={"truthful": 1.0},
        agent_params={"truthful": {"ability": 0.95}},
        mechanism_name="majority_voting",
        mechanism_params={"payment_per_task": 1.0},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42,
    )

    # Run with Majority Voting
    print("Running with Majority Voting...")
    engine_mv = SimulationEngine(config_mv)
    results_mv = engine_mv.run_experiment()

    print("\nMajority Voting Results:")
    print(f"  Accuracy:         {results_mv.mean_metrics['accuracy']:.3f} ± {results_mv.std_metrics['accuracy']:.3f}")
    print(f"  F1 Score:         {results_mv.mean_metrics['f1_score']:.3f} ± {results_mv.std_metrics['f1_score']:.3f}")
    print(f"  Total Payment:    {results_mv.mean_metrics['total_payment']:.2f} ± {results_mv.std_metrics['total_payment']:.2f}")
    print(f"  Quality/Dollar:   {results_mv.mean_metrics['quality_per_dollar']:.4f}")
    print(f"  95% CI Accuracy:  [{results_mv.confidence_intervals['accuracy'][0]:.3f}, {results_mv.confidence_intervals['accuracy'][1]:.3f}]")

    # Run with Dawid-Skene
    config_ds = ExperimentConfig(
        name="exp1a_dawid_skene",
        num_tasks=50,
        num_agents=10,
        agent_mix={"truthful": 1.0},
        agent_params={"truthful": {"ability": 0.95}},
        mechanism_name="dawid_skene",
        mechanism_params={"payment_per_task": 1.0, "quality_weight": 0.5},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42,
    )

    print("\nRunning with Dawid-Skene...")
    engine_ds = SimulationEngine(config_ds)
    results_ds = engine_ds.run_experiment()

    print("\nDawid-Skene Results:")
    print(f"  Accuracy:         {results_ds.mean_metrics['accuracy']:.3f} ± {results_ds.std_metrics['accuracy']:.3f}")
    print(f"  F1 Score:         {results_ds.mean_metrics['f1_score']:.3f} ± {results_ds.std_metrics['f1_score']:.3f}")
    print(f"  Total Payment:    {results_ds.mean_metrics['total_payment']:.2f} ± {results_ds.std_metrics['total_payment']:.2f}")
    print(f"  Quality/Dollar:   {results_ds.mean_metrics['quality_per_dollar']:.4f}")
    print(f"  95% CI Accuracy:  [{results_ds.confidence_intervals['accuracy'][0]:.3f}, {results_ds.confidence_intervals['accuracy'][1]:.3f}]")

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION:")
    mv_pass = results_mv.mean_metrics['accuracy'] > 0.90
    ds_pass = results_ds.mean_metrics['accuracy'] > 0.90
    print(f"  Majority Voting accuracy > 0.90: {'✓ PASS' if mv_pass else '✗ FAIL'}")
    print(f"  Dawid-Skene accuracy > 0.90:     {'✓ PASS' if ds_pass else '✗ FAIL'}")
    print("-" * 70)
    print()

    return mv_pass and ds_pass


def run_experiment_1b():
    """Experiment 1b: All lazy (random) agents.

    Expected: ≈50% accuracy (random baseline) for both mechanisms.
    """
    print("=" * 70)
    print("EXPERIMENT 1b: All lazy (random) agents")
    print("=" * 70)
    print()

    # Configuration
    config_mv = ExperimentConfig(
        name="exp1b_majority_voting",
        num_tasks=100,
        num_agents=10,
        agent_mix={"lazy": 1.0},
        agent_params={},
        mechanism_name="majority_voting",
        mechanism_params={"payment_per_task": 1.0},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42,
    )

    # Run with Majority Voting
    print("Running with Majority Voting...")
    engine_mv = SimulationEngine(config_mv)
    results_mv = engine_mv.run_experiment()

    print("\nMajority Voting Results:")
    print(f"  Accuracy:         {results_mv.mean_metrics['accuracy']:.3f} ± {results_mv.std_metrics['accuracy']:.3f}")
    print(f"  F1 Score:         {results_mv.mean_metrics['f1_score']:.3f} ± {results_mv.std_metrics['f1_score']:.3f}")
    print(f"  Total Payment:    {results_mv.mean_metrics['total_payment']:.2f} ± {results_mv.std_metrics['total_payment']:.2f}")
    print(f"  Quality/Dollar:   {results_mv.mean_metrics['quality_per_dollar']:.4f}")
    print(f"  95% CI Accuracy:  [{results_mv.confidence_intervals['accuracy'][0]:.3f}, {results_mv.confidence_intervals['accuracy'][1]:.3f}]")

    # Run with Dawid-Skene
    config_ds = ExperimentConfig(
        name="exp1b_dawid_skene",
        num_tasks=100,
        num_agents=10,
        agent_mix={"lazy": 1.0},
        agent_params={},
        mechanism_name="dawid_skene",
        mechanism_params={"payment_per_task": 1.0, "quality_weight": 0.5},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42,
    )

    print("\nRunning with Dawid-Skene...")
    engine_ds = SimulationEngine(config_ds)
    results_ds = engine_ds.run_experiment()

    print("\nDawid-Skene Results:")
    print(f"  Accuracy:         {results_ds.mean_metrics['accuracy']:.3f} ± {results_ds.std_metrics['accuracy']:.3f}")
    print(f"  F1 Score:         {results_ds.mean_metrics['f1_score']:.3f} ± {results_ds.std_metrics['f1_score']:.3f}")
    print(f"  Total Payment:    {results_ds.mean_metrics['total_payment']:.2f} ± {results_ds.std_metrics['total_payment']:.2f}")
    print(f"  Quality/Dollar:   {results_ds.mean_metrics['quality_per_dollar']:.4f}")
    print(f"  95% CI Accuracy:  [{results_ds.confidence_intervals['accuracy'][0]:.3f}, {results_ds.confidence_intervals['accuracy'][1]:.3f}]")

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION:")
    mv_pass = 0.40 < results_mv.mean_metrics['accuracy'] < 0.60
    ds_pass = 0.40 < results_ds.mean_metrics['accuracy'] < 0.60
    print(f"  Majority Voting accuracy ≈ 0.50: {'✓ PASS' if mv_pass else '✗ FAIL'}")
    print(f"  Dawid-Skene accuracy ≈ 0.50:     {'✓ PASS' if ds_pass else '✗ FAIL'}")
    print("-" * 70)
    print()

    return mv_pass and ds_pass


def main():
    """Run all Phase 1 validation experiments."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PHASE 1 VALIDATION EXPERIMENTS" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run experiments
    exp1a_passed = run_experiment_1a()
    exp1b_passed = run_experiment_1b()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Experiment 1a (Truthful agents): {'✓ PASSED' if exp1a_passed else '✗ FAILED'}")
    print(f"Experiment 1b (Lazy agents):     {'✓ PASSED' if exp1b_passed else '✗ FAILED'}")
    print()

    if exp1a_passed and exp1b_passed:
        print("🎉 ALL VALIDATION EXPERIMENTS PASSED!")
        print()
        print("Phase 1 is complete. Ready to proceed to Phase 2:")
        print("  - Output Agreement mechanism")
        print("  - RBTS mechanism")
        print("  - Agent prediction methods")
        print("  - Comparative evaluation")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME EXPERIMENTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
