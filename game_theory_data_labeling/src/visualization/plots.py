"""Publication-quality visualization utilities.

This module provides functions to create publication-ready plots for
experimental results. Uses matplotlib with clean, professional styling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8


# Color palette for mechanisms
MECHANISM_COLORS = {
    'majority_voting': '#1f77b4',  # Blue
    'dawid_skene': '#ff7f0e',      # Orange
    'output_agreement': '#2ca02c',  # Green
    'rbts': '#d62728'               # Red
}

MECHANISM_MARKERS = {
    'majority_voting': 'o',
    'dawid_skene': 's',
    'output_agreement': '^',
    'rbts': 'D'
}

MECHANISM_NAMES = {
    'majority_voting': 'Majority Voting',
    'dawid_skene': 'Dawid-Skene',
    'output_agreement': 'Output Agreement',
    'rbts': 'RBTS'
}


def plot_adversarial_robustness(
    results: Dict[str, Dict[int, Dict]],
    output_file: Optional[str] = None
) -> None:
    """Plot accuracy vs adversarial agent percentage.

    Args:
        results: Nested dict {mechanism: {adv_pct: {metrics}}}
        output_file: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for mechanism, adv_results in results.items():
        adv_percentages = sorted(adv_results.keys())
        accuracies = [adv_results[pct]['accuracy'] for pct in adv_percentages]
        std_devs = [adv_results[pct]['accuracy_std'] for pct in adv_percentages]

        ax.plot(
            adv_percentages,
            accuracies,
            marker=MECHANISM_MARKERS.get(mechanism, 'o'),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            linewidth=2,
            markersize=8
        )

        # Add error bars
        ax.fill_between(
            adv_percentages,
            np.array(accuracies) - np.array(std_devs),
            np.array(accuracies) + np.array(std_devs),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            alpha=0.2
        )

    ax.set_xlabel('Adversarial Agents (%)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Robustness to Adversarial Agents', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.close()


def plot_scaling_analysis(
    results: Dict[str, Dict[int, Dict]],
    output_file: Optional[str] = None
) -> None:
    """Plot accuracy and cost-efficiency vs number of agents.

    Args:
        results: Nested dict {mechanism: {num_agents: {metrics}}}
        output_file: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mechanism, agent_results in results.items():
        agent_counts = sorted(agent_results.keys())
        accuracies = [agent_results[n]['accuracy'] for n in agent_counts]
        quality_per_dollar = [agent_results[n]['quality_per_dollar'] for n in agent_counts]

        # Accuracy subplot
        ax1.plot(
            agent_counts,
            accuracies,
            marker=MECHANISM_MARKERS.get(mechanism, 'o'),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            linewidth=2,
            markersize=8
        )

        # Quality per dollar subplot
        ax2.plot(
            agent_counts,
            quality_per_dollar,
            marker=MECHANISM_MARKERS.get(mechanism, 'o'),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            linewidth=2,
            markersize=8
        )

    # Accuracy subplot formatting
    ax1.set_xlabel('Number of Agents', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Agent Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim([0.75, 1.05])

    # Quality per dollar subplot formatting
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Quality per Dollar', fontsize=12)
    ax2.set_title('Cost-Efficiency vs Agent Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.close()


def plot_budget_analysis(
    results: Dict[str, Dict[int, Dict]],
    budget: float,
    output_file: Optional[str] = None
) -> None:
    """Plot accuracy and tasks completed within budget.

    Args:
        results: Nested dict {mechanism: {num_agents: {metrics}}}
        budget: Budget constraint
        output_file: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mechanisms = list(results.keys())
    agent_counts = sorted(results[mechanisms[0]].keys())

    x = np.arange(len(agent_counts))
    width = 0.2

    for i, mechanism in enumerate(mechanisms):
        accuracies = [results[mechanism][n]['accuracy'] for n in agent_counts]
        costs = [results[mechanism][n]['total_payment'] for n in agent_counts]

        # Accuracy bars
        offset = (i - len(mechanisms)/2) * width + width/2
        bars1 = ax1.bar(
            x + offset,
            accuracies,
            width,
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            alpha=0.8
        )

        # Cost bars with budget line
        bars2 = ax2.bar(
            x + offset,
            costs,
            width,
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            color=MECHANISM_COLORS.get(mechanism, 'gray'),
            alpha=0.8
        )

        # Mark bars that exceed budget
        for j, cost in enumerate(costs):
            if cost > budget:
                bars2[j].set_edgecolor('red')
                bars2[j].set_linewidth(2)

    # Budget line
    ax2.axhline(y=budget, color='red', linestyle='--', linewidth=2, label=f'Budget (${budget:.0f})')

    # Accuracy subplot formatting
    ax1.set_xlabel('Number of Agents', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy by Agent Count', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agent_counts)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim([0.75, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Cost subplot formatting
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Total Cost ($)', fontsize=12)
    ax2.set_title('Cost vs Budget Constraint', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_counts)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.close()


def plot_difficulty_comparison(
    results: Dict[str, Dict[str, Dict]],
    output_file: Optional[str] = None
) -> None:
    """Plot mechanism performance on easy vs hard tasks.

    Args:
        results: Nested dict {mechanism: {difficulty: {metrics}}}
        output_file: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mechanisms = list(results.keys())
    difficulties = ['easy', 'hard']

    x = np.arange(len(mechanisms))
    width = 0.35

    easy_accuracies = [results[m]['easy']['accuracy'] for m in mechanisms]
    hard_accuracies = [results[m]['hard']['accuracy'] for m in mechanisms]

    easy_qpd = [results[m]['easy']['quality_per_dollar'] for m in mechanisms]
    hard_qpd = [results[m]['hard']['quality_per_dollar'] for m in mechanisms]

    # Accuracy comparison
    bars1 = ax1.bar(
        x - width/2,
        easy_accuracies,
        width,
        label='Easy Tasks',
        color='skyblue',
        alpha=0.8
    )
    bars2 = ax1.bar(
        x + width/2,
        hard_accuracies,
        width,
        label='Hard Tasks',
        color='coral',
        alpha=0.8
    )

    # Quality per dollar comparison
    bars3 = ax2.bar(
        x - width/2,
        easy_qpd,
        width,
        label='Easy Tasks',
        color='skyblue',
        alpha=0.8
    )
    bars4 = ax2.bar(
        x + width/2,
        hard_qpd,
        width,
        label='Hard Tasks',
        color='coral',
        alpha=0.8
    )

    # Accuracy subplot formatting
    ax1.set_xlabel('Mechanism', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy: Easy vs Hard Tasks', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MECHANISM_NAMES.get(m, m) for m in mechanisms], rotation=15, ha='right')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim([0.75, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Quality per dollar subplot formatting
    ax2.set_xlabel('Mechanism', fontsize=12)
    ax2.set_ylabel('Quality per Dollar', fontsize=12)
    ax2.set_title('Cost-Efficiency: Easy vs Hard Tasks', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MECHANISM_NAMES.get(m, m) for m in mechanisms], rotation=15, ha='right')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.close()


def plot_mechanism_comparison_summary(
    mechanism_metrics: Dict[str, Dict[str, float]],
    output_file: Optional[str] = None
) -> None:
    """Create a radar chart comparing mechanisms across multiple dimensions.

    Args:
        mechanism_metrics: Dict {mechanism: {metric_name: normalized_value}}
        output_file: Path to save figure (optional)
    """
    # Prepare data
    mechanisms = list(mechanism_metrics.keys())
    metrics = list(mechanism_metrics[mechanisms[0]].keys())
    num_metrics = len(metrics)

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for mechanism in mechanisms:
        values = [mechanism_metrics[mechanism][m] for m in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            label=MECHANISM_NAMES.get(mechanism, mechanism),
            color=MECHANISM_COLORS.get(mechanism, 'gray')
        )
        ax.fill(
            angles,
            values,
            alpha=0.15,
            color=MECHANISM_COLORS.get(mechanism, 'gray')
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Mechanism Comparison Across Dimensions', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.close()


def create_all_visualizations(output_dir: str = "results/figures") -> None:
    """Create all publication-quality visualizations.

    Args:
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Visualization suite would create plots in: {output_path}")
    print("Note: Run experiments first to generate data for plotting.")
    print("\nAvailable plot functions:")
    print("  - plot_adversarial_robustness()")
    print("  - plot_scaling_analysis()")
    print("  - plot_budget_analysis()")
    print("  - plot_difficulty_comparison()")
    print("  - plot_mechanism_comparison_summary()")


if __name__ == "__main__":
    create_all_visualizations()
