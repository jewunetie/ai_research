"""Experiment 4b: Scaling Study

Research Question: How does performance scale with the number of agents?

Design:
- Vary number of agents: 3, 5, 10, 20, 50
- Agent pool: Mixed quality (70% truthful ability=0.85, 30% lazy)
- 50 tasks
- 10 runs per configuration
- Compare accuracy and cost-efficiency as agent count increases

Metrics:
- Accuracy vs number of agents
- Cost per task vs number of agents
- Quality per dollar vs number of agents
- Marginal improvement in accuracy
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine
from src.utils.reproducibility import save_results_summary


def run_scaling_experiment(mechanism_name: str, num_agents: int):
    """Run scaling experiment for a mechanism.

    Args:
        mechanism_name: Name of mechanism to test
        num_agents: Number of agents to use

    Returns:
        Results from experiment run
    """
    config = ExperimentConfig(
        name=f"exp4b_{mechanism_name}_n{num_agents}",
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

    # Calculate per-task metrics
    cost_per_task = results.mean_metrics['total_payment'] / config.num_tasks

    return {
        "accuracy": results.mean_metrics['accuracy'],
        "accuracy_std": results.std_metrics['accuracy'],
        "total_payment": results.mean_metrics['total_payment'],
        "payment_std": results.std_metrics['total_payment'],
        "cost_per_task": cost_per_task,
        "quality_per_dollar": results.mean_metrics['accuracy'] / results.mean_metrics['total_payment'],
        "num_runs": config.num_runs
    }


def main():
    """Run Experiment 4b: Scaling Study."""
    mechanisms = ["majority_voting", "dawid_skene", "output_agreement", "rbts"]
    agent_counts = [3, 5, 10, 20, 50]

    print("=" * 80)
    print("EXPERIMENT 4b: SCALING STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Tasks: 50")
    print(f"  Agent counts: {agent_counts}")
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

            results = run_scaling_experiment(mechanism, num_agents)
            all_results[mechanism][num_agents] = results

            print(f"Accuracy: {results['accuracy']:.1%}, "
                  f"Cost/task: ${results['cost_per_task']:.2f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ACCURACY SCALING")
    print("=" * 80)
    print()

    print(f"{'Mechanism':<20} ", end="")
    for num_agents in agent_counts:
        print(f"{num_agents:>7} agents", end="  ")
    print()
    print("-" * 80)

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display:<20} ", end="")
        for num_agents in agent_counts:
            accuracy = all_results[mechanism][num_agents]['accuracy']
            print(f"   {accuracy:>6.1%}   ", end="")
        print()

    # Cost per task scaling
    print("\n" + "=" * 80)
    print("COST PER TASK SCALING")
    print("=" * 80)
    print()

    print(f"{'Mechanism':<20} ", end="")
    for num_agents in agent_counts:
        print(f"{num_agents:>7} agents", end="  ")
    print()
    print("-" * 80)

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display:<20} ", end="")
        for num_agents in agent_counts:
            cost = all_results[mechanism][num_agents]['cost_per_task']
            print(f"   ${cost:>6.2f}  ", end="")
        print()

    # Quality per dollar scaling
    print("\n" + "=" * 80)
    print("QUALITY PER DOLLAR SCALING")
    print("=" * 80)
    print()

    print(f"{'Mechanism':<20} ", end="")
    for num_agents in agent_counts:
        print(f"{num_agents:>7} agents", end="  ")
    print()
    print("-" * 80)

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display:<20} ", end="")
        for num_agents in agent_counts:
            qpd = all_results[mechanism][num_agents]['quality_per_dollar']
            print(f" {qpd:>8.5f} ", end="")
        print()

    # Analyze marginal improvements
    print("\n" + "=" * 80)
    print("MARGINAL ACCURACY IMPROVEMENT")
    print("=" * 80)
    print()

    print("When increasing agents from N to 2N:")
    print()

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display}:")

        # Compare 3->5, 5->10, 10->20, 20->50
        pairs = [(3, 5), (5, 10), (10, 20), (20, 50)]
        for n1, n2 in pairs:
            if n1 in agent_counts and n2 in agent_counts:
                acc1 = all_results[mechanism][n1]['accuracy']
                acc2 = all_results[mechanism][n2]['accuracy']
                improvement = (acc2 - acc1) * 100  # percentage points
                print(f"  {n1:2d} → {n2:2d} agents: {improvement:+.1f} pp")
        print()

    # Find optimal agent count for each mechanism
    print("=" * 80)
    print("OPTIMAL AGENT COUNT BY CRITERION")
    print("=" * 80)
    print()

    for mechanism in mechanisms:
        mech_display = mechanism.replace("_", " ").title()
        print(f"{mech_display}:")

        # Best accuracy
        best_acc = 0.0
        best_acc_count = 0
        for num_agents in agent_counts:
            acc = all_results[mechanism][num_agents]['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_acc_count = num_agents

        # Best quality/dollar
        best_qpd = 0.0
        best_qpd_count = 0
        for num_agents in agent_counts:
            qpd = all_results[mechanism][num_agents]['quality_per_dollar']
            if qpd > best_qpd:
                best_qpd = qpd
                best_qpd_count = num_agents

        print(f"  Best accuracy: {best_acc_count} agents ({best_acc:.1%})")
        print(f"  Best quality/$: {best_qpd_count} agents ({best_qpd:.5f})")
        print()

    # Save results
    output_file = Path(__file__).parent / "results" / "experiment_4b_scaling.json"
    output_file.parent.mkdir(exist_ok=True)

    save_results_summary(
        {"experiment_4b_scaling_study": all_results},
        output_file=str(output_file)
    )

    print(f"Results saved to: {output_file}")
    print("\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    main()
