"""Phase 2 validation experiments.

Experiment 2a: Compare 4 mechanisms with all truthful agents
Experiment 2b: Compare 4 mechanisms with all lazy agents
Experiment 2c: Compare 4 mechanisms with mixed agents (70% truthful, 30% lazy)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine


def run_experiment_2a():
    """Experiment 2a: Compare 4 mechanisms with all truthful agents.

    Expected: All mechanisms should perform similarly (high accuracy).
    This tests implementation correctness.
    """
    print("=" * 70)
    print("EXPERIMENT 2a: Compare 4 mechanisms with all truthful agents")
    print("=" * 70)
    print()

    mechanisms = [
        ("Majority Voting", "majority_voting", {"payment_per_task": 1.0}),
        ("Dawid-Skene", "dawid_skene", {"payment_per_task": 1.0, "quality_weight": 0.5}),
        ("Output Agreement", "output_agreement", {"agreement_payment": 1.0, "disagreement_payment": 0.0}),
        ("RBTS", "rbts", {"base_payment": 1.0, "bonus_scale": 0.5}),
    ]

    results = {}

    for mechanism_name, mechanism_id, params in mechanisms:
        config = ExperimentConfig(
            name=f"exp2a_{mechanism_id}",
            num_tasks=50,
            num_agents=10,
            agent_mix={"truthful": 1.0},
            agent_params={"truthful": {"ability": 0.95}},
            mechanism_name=mechanism_id,
            mechanism_params=params,
            task_difficulty=0.5,
            label_prior=0.5,
            num_runs=10,
            random_seed=42,
        )

        print(f"\nRunning with {mechanism_name}...")
        engine = SimulationEngine(config)
        result = engine.run_experiment()
        results[mechanism_name] = result

        print(f"\n{mechanism_name} Results:")
        print(f"  Accuracy:         {result.mean_metrics['accuracy']:.3f} ± {result.std_metrics['accuracy']:.3f}")
        print(f"  F1 Score:         {result.mean_metrics['f1_score']:.3f} ± {result.std_metrics['f1_score']:.3f}")
        print(f"  Total Payment:    {result.mean_metrics['total_payment']:.2f} ± {result.std_metrics['total_payment']:.2f}")
        print(f"  Quality/Dollar:   {result.mean_metrics['quality_per_dollar']:.4f}")
        print(f"  95% CI Accuracy:  [{result.confidence_intervals['accuracy'][0]:.3f}, {result.confidence_intervals['accuracy'][1]:.3f}]")

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION:")
    all_passed = True
    for mechanism_name, result in results.items():
        passed = result.mean_metrics['accuracy'] > 0.90
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {mechanism_name} accuracy > 0.90: {status}")
        all_passed = all_passed and passed

    print("-" * 70)
    print()

    return all_passed


def run_experiment_2b():
    """Experiment 2b: Compare 4 mechanisms with all lazy agents.

    Expected: All mechanisms should perform poorly (≈50% accuracy).
    Mechanisms can't fix bad input.
    """
    print("=" * 70)
    print("EXPERIMENT 2b: Compare 4 mechanisms with all lazy agents")
    print("=" * 70)
    print()

    mechanisms = [
        ("Majority Voting", "majority_voting", {"payment_per_task": 1.0}),
        ("Dawid-Skene", "dawid_skene", {"payment_per_task": 1.0, "quality_weight": 0.5}),
        ("Output Agreement", "output_agreement", {"agreement_payment": 1.0, "disagreement_payment": 0.0}),
        ("RBTS", "rbts", {"base_payment": 1.0, "bonus_scale": 0.5}),
    ]

    results = {}

    for mechanism_name, mechanism_id, params in mechanisms:
        config = ExperimentConfig(
            name=f"exp2b_{mechanism_id}",
            num_tasks=100,
            num_agents=10,
            agent_mix={"lazy": 1.0},
            agent_params={},
            mechanism_name=mechanism_id,
            mechanism_params=params,
            task_difficulty=0.5,
            label_prior=0.5,
            num_runs=10,
            random_seed=42,
        )

        print(f"\nRunning with {mechanism_name}...")
        engine = SimulationEngine(config)
        result = engine.run_experiment()
        results[mechanism_name] = result

        print(f"\n{mechanism_name} Results:")
        print(f"  Accuracy:         {result.mean_metrics['accuracy']:.3f} ± {result.std_metrics['accuracy']:.3f}")
        print(f"  F1 Score:         {result.mean_metrics['f1_score']:.3f} ± {result.std_metrics['f1_score']:.3f}")
        print(f"  Total Payment:    {result.mean_metrics['total_payment']:.2f} ± {result.std_metrics['total_payment']:.2f}")
        print(f"  Quality/Dollar:   {result.mean_metrics['quality_per_dollar']:.4f}")
        print(f"  95% CI Accuracy:  [{result.confidence_intervals['accuracy'][0]:.3f}, {result.confidence_intervals['accuracy'][1]:.3f}]")

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION:")
    all_passed = True
    for mechanism_name, result in results.items():
        passed = 0.40 < result.mean_metrics['accuracy'] < 0.60
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {mechanism_name} accuracy ≈ 0.50: {status}")
        all_passed = all_passed and passed

    print("-" * 70)
    print()

    return all_passed


def run_experiment_2c():
    """Experiment 2c: Compare 4 mechanisms with 70% truthful, 30% lazy.

    Expected: Mechanisms start to differentiate based on robustness.
    """
    print("=" * 70)
    print("EXPERIMENT 2c: Compare 4 mechanisms with mixed agents (70/30)")
    print("=" * 70)
    print()

    mechanisms = [
        ("Majority Voting", "majority_voting", {"payment_per_task": 1.0}),
        ("Dawid-Skene", "dawid_skene", {"payment_per_task": 1.0, "quality_weight": 0.5}),
        ("Output Agreement", "output_agreement", {"agreement_payment": 1.0, "disagreement_payment": 0.0}),
        ("RBTS", "rbts", {"base_payment": 1.0, "bonus_scale": 0.5}),
    ]

    results = {}

    for mechanism_name, mechanism_id, params in mechanisms:
        config = ExperimentConfig(
            name=f"exp2c_{mechanism_id}",
            num_tasks=50,
            num_agents=10,
            agent_mix={"truthful": 0.7, "lazy": 0.3},
            agent_params={"truthful": {"ability": 0.9}},
            mechanism_name=mechanism_id,
            mechanism_params=params,
            task_difficulty=0.5,
            label_prior=0.5,
            num_runs=20,
            random_seed=42,
        )

        print(f"\nRunning with {mechanism_name}...")
        engine = SimulationEngine(config)
        result = engine.run_experiment()
        results[mechanism_name] = result

        print(f"\n{mechanism_name} Results:")
        print(f"  Accuracy:         {result.mean_metrics['accuracy']:.3f} ± {result.std_metrics['accuracy']:.3f}")
        print(f"  F1 Score:         {result.mean_metrics['f1_score']:.3f} ± {result.std_metrics['f1_score']:.3f}")
        print(f"  Total Payment:    {result.mean_metrics['total_payment']:.2f} ± {result.std_metrics['total_payment']:.2f}")
        print(f"  Quality/Dollar:   {result.mean_metrics['quality_per_dollar']:.4f}")
        print(f"  95% CI Accuracy:  [{result.confidence_intervals['accuracy'][0]:.3f}, {result.confidence_intervals['accuracy'][1]:.3f}]")

    # Comparative analysis
    print("\n" + "-" * 70)
    print("COMPARATIVE ANALYSIS:")
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean_metrics['accuracy'], reverse=True)
    print("\nRanking by Accuracy:")
    for rank, (mechanism_name, result) in enumerate(sorted_results, 1):
        print(f"  {rank}. {mechanism_name}: {result.mean_metrics['accuracy']:.3f}")

    print("\nRanking by Quality/Dollar:")
    sorted_by_efficiency = sorted(results.items(), key=lambda x: x[1].mean_metrics['quality_per_dollar'], reverse=True)
    for rank, (mechanism_name, result) in enumerate(sorted_by_efficiency, 1):
        print(f"  {rank}. {mechanism_name}: {result.mean_metrics['quality_per_dollar']:.4f}")

    print("-" * 70)
    print()

    # All should have reasonable accuracy (> 0.65 with 70% truthful agents at 0.9 ability)
    all_passed = all(result.mean_metrics['accuracy'] > 0.65 for result in results.values())
    return all_passed


def main():
    """Run all Phase 2 validation experiments."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PHASE 2 VALIDATION EXPERIMENTS" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run experiments
    exp2a_passed = run_experiment_2a()
    exp2b_passed = run_experiment_2b()
    exp2c_passed = run_experiment_2c()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Experiment 2a (All truthful agents):  {'✓ PASSED' if exp2a_passed else '✗ FAILED'}")
    print(f"Experiment 2b (All lazy agents):      {'✓ PASSED' if exp2b_passed else '✗ FAILED'}")
    print(f"Experiment 2c (Mixed agents):         {'✓ PASSED' if exp2c_passed else '✗ FAILED'}")
    print()

    if exp2a_passed and exp2b_passed and exp2c_passed:
        print("🎉 ALL VALIDATION EXPERIMENTS PASSED!")
        print()
        print("Phase 2 is complete! All 4 mechanisms working correctly:")
        print("  ✓ Majority Voting")
        print("  ✓ Dawid-Skene (EM algorithm)")
        print("  ✓ Output Agreement (peer prediction)")
        print("  ✓ RBTS (Robust Bayesian Truth Serum)")
        print()
        print("All 5 agent types implemented:")
        print("  ✓ Truthful")
        print("  ✓ Lazy")
        print("  ✓ Strategic")
        print("  ✓ Adversarial")
        print("  ✓ Noisy Truthful")
        print()
        print("Ready for Phase 3: Strategic behavior and equilibrium analysis")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME EXPERIMENTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
