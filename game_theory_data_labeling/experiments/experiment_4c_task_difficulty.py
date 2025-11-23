"""Experiment 4c: Task Difficulty

Research Question: How do mechanisms perform on easy vs hard tasks?

Design:
- Vary task difficulty: Easy (agents have 0.95 ability), Hard (agents have 0.70 ability)
- 10 agents
- Agent pool: 100% truthful agents with varying ability
- 50 tasks
- 15 runs per configuration
- Compare mechanism performance on easy vs hard tasks

Metrics:
- Accuracy on easy vs hard tasks
- Payment difference between easy and hard
- Which mechanisms adapt better to task difficulty
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine
from src.utils.reproducibility import save_results_summary
from src.evaluation.statistical_tests import paired_t_test


def run_difficulty_experiment(mechanism_name: str, agent_ability: float, difficulty_label: str):
    """Run task difficulty experiment for a mechanism.

    Args:
        mechanism_name: Name of mechanism to test
        agent_ability: Ability level for agents (higher = easier tasks)
        difficulty_label: Label for this difficulty level

    Returns:
        Results from experiment run
    """
    config = ExperimentConfig(
        name=f"exp4c_{mechanism_name}_{difficulty_label}",
        num_tasks=50,
        num_agents=10,
        agent_mix={"truthful": 1.0},
        agent_params={"truthful": {"ability": agent_ability}},
        mechanism_name=mechanism_name,
        mechanism_params={"payment_per_task": 1.0},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=15,
        random_seed=42,
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    return {
        "difficulty": difficulty_label,
        "agent_ability": agent_ability,
        "accuracy": results.mean_metrics['accuracy'],
        "accuracy_std": results.std_metrics['accuracy'],
        "accuracy_ci": results.confidence_intervals['accuracy'],
        "total_payment": results.mean_metrics['total_payment'],
        "payment_std": results.std_metrics['total_payment'],
        "quality_per_dollar": results.mean_metrics['accuracy'] / results.mean_metrics['total_payment'],
        "num_runs": config.num_runs,
        # Store raw accuracies for statistical testing
        "raw_accuracies": [metric_result.accuracy for metric_result in results.metrics]
    }


def main():
    """Run Experiment 4c: Task Difficulty."""
    mechanisms = ["majority_voting", "dawid_skene", "output_agreement", "rbts"]

    # Define difficulty levels
    difficulty_levels = [
        (0.95, "easy"),
        (0.70, "hard")
    ]

    print("=" * 80)
    print("EXPERIMENT 4c: TASK DIFFICULTY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Tasks: 50")
    print(f"  Agents: 10 (all truthful)")
    print(f"  Difficulty levels:")
    print(f"    - Easy: agent ability = 0.95")
    print(f"    - Hard: agent ability = 0.70")
    print(f"  Runs per configuration: 15")
    print()

    # Store all results
    all_results = {}

    # Run experiments for each mechanism and difficulty level
    for mechanism in mechanisms:
        print(f"\nTesting {mechanism}...")
        all_results[mechanism] = {}

        for ability, difficulty in difficulty_levels:
            print(f"  {difficulty.capitalize()} tasks (ability={ability})... ", end="", flush=True)

            results = run_difficulty_experiment(mechanism, ability, difficulty)
            all_results[mechanism][difficulty] = results

            print(f"Accuracy: {results['accuracy']:.1%} ± {results['accuracy_std']:.1%}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("EASY vs HARD TASK COMPARISON")
    print("=" * 80)

    for difficulty_label in ["easy", "hard"]:
        print(f"\n--- {difficulty_label.upper()} TASKS ---\n")
        print(f"{'Mechanism':<20} {'Accuracy':<15} {'Payment':<12} {'Quality/$':<12}")
        print("-" * 70)

        for mechanism in mechanisms:
            results = all_results[mechanism][difficulty_label]
            mech_display = mechanism.replace("_", " ").title()
            ci_low, ci_high = results['accuracy_ci']

            print(f"{mech_display:<20} "
                  f"{results['accuracy']:>6.1%} [{ci_low:.1%}, {ci_high:.1%}]  "
                  f"${results['total_payment']:>9.2f}  "
                  f"{results['quality_per_dollar']:>10.6f}")

    # Statistical significance tests
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Easy vs Hard)")
    print("=" * 80)
    print()

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display}:")

        easy_accuracies = np.array(all_results[mechanism]["easy"]["raw_accuracies"])
        hard_accuracies = np.array(all_results[mechanism]["hard"]["raw_accuracies"])

        # Paired t-test
        test_result = paired_t_test(easy_accuracies, hard_accuracies, alpha=0.05)

        print(f"  Mean difference: {test_result.mean_diff:+.1%}")
        print(f"  p-value: {test_result.p_value:.4f}")
        print(f"  Cohen's d: {test_result.cohen_d:.3f}")
        print(f"  {test_result.interpretation}")
        print()

    # Adaptability analysis
    print("=" * 80)
    print("MECHANISM ADAPTABILITY TO TASK DIFFICULTY")
    print("=" * 80)
    print()

    print(f"{'Mechanism':<20} {'Easy Accuracy':<15} {'Hard Accuracy':<15} {'Drop':<10} {'Relative Drop'}")
    print("-" * 80)

    adaptability_scores = []

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        easy_acc = all_results[mechanism]["easy"]["accuracy"]
        hard_acc = all_results[mechanism]["hard"]["accuracy"]

        drop = easy_acc - hard_acc
        relative_drop = drop / easy_acc if easy_acc > 0 else 0

        adaptability_scores.append((mechanism, relative_drop))

        print(f"{mech_display:<20} "
              f"{easy_acc:>13.1%}  "
              f"{hard_acc:>13.1%}  "
              f"{drop:>8.1%}  "
              f"{relative_drop:>13.1%}")

    # Find most robust mechanism (smallest relative drop)
    adaptability_scores.sort(key=lambda x: x[1])
    best_mechanism = adaptability_scores[0][0].replace("_", " ").title()
    best_drop = adaptability_scores[0][1]

    print()
    print(f"Most robust to task difficulty: {best_mechanism}")
    print(f"  (only {best_drop:.1%} relative accuracy drop)")

    # Cost-efficiency comparison
    print("\n" + "=" * 80)
    print("COST-EFFICIENCY BY DIFFICULTY")
    print("=" * 80)
    print()

    print(f"{'Mechanism':<20} {'Easy Q/$':<15} {'Hard Q/$':<15} {'Change'}")
    print("-" * 65)

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        easy_qpd = all_results[mechanism]["easy"]["quality_per_dollar"]
        hard_qpd = all_results[mechanism]["hard"]["quality_per_dollar"]

        change = ((hard_qpd - easy_qpd) / easy_qpd * 100) if easy_qpd > 0 else 0

        print(f"{mech_display:<20} "
              f"{easy_qpd:>13.6f}  "
              f"{hard_qpd:>13.6f}  "
              f"{change:>+10.1f}%")

    # Save results
    output_file = Path(__file__).parent / "results" / "experiment_4c_difficulty.json"
    output_file.parent.mkdir(exist_ok=True)

    save_results_summary(
        {"experiment_4c_task_difficulty": all_results},
        output_file=str(output_file)
    )

    print()
    print(f"Results saved to: {output_file}")
    print("\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    main()
