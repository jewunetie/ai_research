"""Experiment 4a: Budget Constraints

Research Question: With a fixed budget, which mechanism achieves the best quality?

Design:
- Fixed budget: $500
- Vary number of agents: 5, 10, 15, 20
- Agent pool: Mixed quality (70% truthful ability=0.85, 30% lazy)
- 50 tasks
- 10 runs per configuration
- Compare quality achieved by each mechanism within budget

Metrics:
- Final accuracy
- Total cost
- Quality per dollar
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine
from src.utils.reproducibility import save_results_summary


def run_budget_experiment(mechanism_name: str, num_agents: int, budget: float = 500.0):
    """Run budget-constrained experiment for a mechanism.

    Args:
        mechanism_name: Name of mechanism to test
        num_agents: Number of agents to use
        budget: Maximum budget allowed

    Returns:
        Results from experiment run
    """
    config = ExperimentConfig(
        name=f"exp4a_{mechanism_name}_n{num_agents}",
        num_tasks=50,
        num_agents=num_agents,
        agent_mix={"truthful": 0.7, "lazy": 0.3},
        agent_params={"truthful": {"ability": 0.85}},
        mechanism_name=mechanism_name,
        mechanism_params={"payment_per_task": 1.0},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42,
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Check if within budget
    mean_payment = results.mean_metrics['total_payment']
    within_budget = mean_payment <= budget

    return {
        "accuracy": results.mean_metrics['accuracy'],
        "accuracy_std": results.std_metrics['accuracy'],
        "total_payment": mean_payment,
        "payment_std": results.std_metrics['total_payment'],
        "quality_per_dollar": results.mean_metrics['accuracy'] / mean_payment if mean_payment > 0 else 0.0,
        "within_budget": within_budget,
        "num_runs": config.num_runs
    }


def main():
    """Run Experiment 4a: Budget Constraints."""
    mechanisms = ["majority_voting", "dawid_skene", "output_agreement", "rbts"]
    agent_counts = [5, 10, 15, 20]
    budget = 500.0

    print("=" * 80)
    print("EXPERIMENT 4a: BUDGET CONSTRAINTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Budget: ${budget:.2f}")
    print(f"  Tasks: 50")
    print(f"  Agent counts tested: {agent_counts}")
    print(f"  Agent mix: 70% truthful (ability=0.85), 30% lazy")
    print(f"  Runs per configuration: 10")
    print()

    # Store all results
    all_results = {}

    # Run experiments for each mechanism and agent count
    for mechanism in mechanisms:
        print(f"\nTesting {mechanism}...")
        all_results[mechanism] = {}

        for num_agents in agent_counts:
            print(f"  {num_agents} agents... ", end="", flush=True)

            results = run_budget_experiment(mechanism, num_agents, budget)
            all_results[mechanism][num_agents] = results

            status = "✓" if results['within_budget'] else "✗ OVER BUDGET"
            print(f"Accuracy: {results['accuracy']:.1%}, "
                  f"Cost: ${results['total_payment']:.2f} {status}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for num_agents in agent_counts:
        print(f"\n--- {num_agents} Agents (Budget: ${budget:.2f}) ---\n")
        print(f"{'Mechanism':<20} {'Accuracy':<12} {'Cost':<12} {'Quality/$':<12} {'Budget OK'}")
        print("-" * 75)

        for mechanism in mechanisms:
            results = all_results[mechanism][num_agents]
            budget_ok = "✓" if results['within_budget'] else "✗"
            mech_display = mechanism.replace("_", " ").title()

            print(f"{mech_display:<20} "
                  f"{results['accuracy']:>10.1%}  "
                  f"${results['total_payment']:>9.2f}  "
                  f"{results['quality_per_dollar']:>10.6f}  "
                  f"{budget_ok}")

    # Find best mechanism for each configuration
    print("\n" + "=" * 80)
    print("BEST MECHANISM BY AGENT COUNT (WITHIN BUDGET)")
    print("=" * 80)
    print()

    for num_agents in agent_counts:
        best_accuracy = 0.0
        best_quality_per_dollar = 0.0
        best_mech_accuracy = ""
        best_mech_quality = ""

        for mechanism in mechanisms:
            results = all_results[mechanism][num_agents]

            # Only consider if within budget
            if not results['within_budget']:
                continue

            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_mech_accuracy = mechanism.replace("_", " ").title()

            if results['quality_per_dollar'] > best_quality_per_dollar:
                best_quality_per_dollar = results['quality_per_dollar']
                best_mech_quality = mechanism.replace("_", " ").title()

        if best_mech_accuracy:
            print(f"{num_agents} agents:")
            print(f"  Best Accuracy: {best_mech_accuracy} ({best_accuracy:.1%})")
            print(f"  Best Quality/$: {best_mech_quality} ({best_quality_per_dollar:.6f})")
        else:
            print(f"{num_agents} agents: No mechanisms within budget")
        print()

    # Save results
    output_file = Path(__file__).parent / "results" / "experiment_4a_budget.json"
    output_file.parent.mkdir(exist_ok=True)

    save_results_summary(
        {"experiment_4a_budget_constraints": all_results},
        output_file=str(output_file)
    )

    print(f"Results saved to: {output_file}")
    print("\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    main()
