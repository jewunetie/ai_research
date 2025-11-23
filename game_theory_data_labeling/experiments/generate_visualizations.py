"""Generate all publication-quality visualizations from experiment results.

This script loads results from Phase 3 and Phase 4 experiments and creates
publication-ready figures.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.plots import (
    plot_adversarial_robustness,
    plot_scaling_analysis,
    plot_budget_analysis,
    plot_difficulty_comparison
)


def load_results(results_dir: Path) -> dict:
    """Load all experiment results from JSON files.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dictionary with all loaded results
    """
    results = {}

    # Load each result file
    result_files = {
        'scaling': 'experiment_4b_scaling.json',
        'budget': 'experiment_4a_budget.json',
        'difficulty': 'experiment_4c_difficulty.json',
    }

    for key, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Extract the nested experiment data
                if 'experiments' in data:
                    # Get first (and only) experiment
                    exp_data = list(data['experiments'].values())[0]
                    results[key] = exp_data
                else:
                    results[key] = data
        else:
            print(f"Warning: {filepath} not found, skipping {key} visualization")

    return results


def generate_all_plots(output_dir: str = "results/figures"):
    """Generate all visualizations from experiment results.

    Args:
        output_dir: Directory to save generated plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load results
    results_dir = Path(__file__).parent / "results"
    results = load_results(results_dir)

    print("=" * 80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 80)
    print()

    # 1. Scaling Analysis
    if 'scaling' in results:
        print("Creating scaling analysis plots...")
        scaling_data = results['scaling']

        # Convert string keys to integers for agent counts
        formatted_scaling = {}
        for mechanism, agent_results in scaling_data.items():
            formatted_scaling[mechanism] = {}
            for agent_count_str, metrics in agent_results.items():
                agent_count = int(agent_count_str)
                formatted_scaling[mechanism][agent_count] = metrics

        plot_scaling_analysis(
            formatted_scaling,
            output_file=str(output_path / "scaling_analysis.png")
        )
        print("  ✓ Saved: scaling_analysis.png")
    else:
        print("  ✗ Skipping scaling analysis (data not found)")

    # 2. Budget Analysis
    if 'budget' in results:
        print("\nCreating budget constraint plots...")
        budget_data = results['budget']

        # Convert string keys to integers
        formatted_budget = {}
        for mechanism, agent_results in budget_data.items():
            formatted_budget[mechanism] = {}
            for agent_count_str, metrics in agent_results.items():
                agent_count = int(agent_count_str)
                formatted_budget[mechanism][agent_count] = metrics

        plot_budget_analysis(
            formatted_budget,
            budget=500.0,
            output_file=str(output_path / "budget_analysis.png")
        )
        print("  ✓ Saved: budget_analysis.png")
    else:
        print("  ✗ Skipping budget analysis (data not found)")

    # 3. Task Difficulty Comparison
    if 'difficulty' in results:
        print("\nCreating task difficulty comparison plots...")
        difficulty_data = results['difficulty']

        plot_difficulty_comparison(
            difficulty_data,
            output_file=str(output_path / "difficulty_comparison.png")
        )
        print("  ✓ Saved: difficulty_comparison.png")
    else:
        print("  ✗ Skipping difficulty comparison (data not found)")

    # 4. Adversarial Robustness (if Phase 3b data exists)
    # Check if we have Phase 3 results with adversarial data
    phase3_results = Path(__file__).parent.parent / "results" / "phase3"
    if phase3_results.exists():
        # Look for adversarial experiment results
        adv_files = list(phase3_results.glob("experiment_3b*.json"))
        if adv_files:
            print("\nCreating adversarial robustness plots...")
            # Load and plot adversarial data
            # (This would require the actual Phase 3b results structure)
            print("  ℹ Adversarial robustness data available in Phase 3 results")
        else:
            print("\n  ℹ Note: Run Phase 3b experiment for adversarial robustness plots")
    else:
        print("\n  ℹ Note: Phase 3 results not found")

    print()
    print("=" * 80)
    print(f"All available visualizations saved to: {output_path.absolute()}")
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    generate_all_plots()


if __name__ == "__main__":
    main()
