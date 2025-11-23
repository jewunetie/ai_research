"""Main entry point for the labor-for-content-access simulation"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_loader import load_config, validate_config
from src.simulation.model import LaborContentAccessModel


def main(config_path=None):
    """
    Main function to run the simulation

    Args:
        config_path: Optional path to config file
    """
    print("=" * 70)
    print("Labor-for-Content-Access Simulation")
    print("=" * 70)

    # Load configuration
    print("\n[1/5] Loading configuration...")
    config = load_config(config_path)
    validate_config(config)
    print(f"  Config loaded successfully")
    print(f"  Random seed: {config['simulation']['random_seed']}")
    print(f"  Num steps: {config['simulation']['num_steps']}")

    # Create model
    print("\n[2/5] Initializing simulation model...")
    model = LaborContentAccessModel(config)
    print(f"  Model initialized with {len(model.users)} users, "
          f"{len(model.creators)} creators, {len(model.companies)} companies")

    # Run simulation
    print("\n[3/5] Running simulation...")
    num_steps = config['simulation']['num_steps']
    model.run_simulation(num_steps)

    # Generate reports
    print("\n[4/5] Generating reports...")
    summary = model.get_summary_report()

    # Display summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    print("\n## Choice Distribution")
    choice_dist = summary['metrics']['summary']['choice_distribution']
    print(f"  Labor:   {choice_dist['labor_count']:6d} ({choice_dist['labor_pct']:.1f}%)")
    print(f"  Ads:     {choice_dist['ads_count']:6d} ({choice_dist['ads_pct']:.1f}%)")
    print(f"  Payment: {choice_dist['payment_count']:6d} ({choice_dist['payment_pct']:.1f}%)")

    print("\n## Revenue")
    revenue = summary['revenue']
    print(f"  Total Revenue:    ${revenue['total_revenue']:.2f}")
    print(f"  Labor Revenue:    ${revenue['revenue_by_source']['labor']:.2f} "
          f"({revenue['labor_percentage']:.1f}%)")
    print(f"  Ads Revenue:      ${revenue['revenue_by_source']['ads']:.2f} "
          f"({revenue['ads_percentage']:.1f}%)")
    print(f"  Payment Revenue:  ${revenue['revenue_by_source']['payment']:.2f} "
          f"({revenue['payment_percentage']:.1f}%)")

    print("\n## Quality Metrics")
    quality = summary['quality']
    print(f"  Avg Fleiss' Kappa: {quality['avg_fleiss_kappa']:.3f}")
    print(f"  Avg Accuracy:      {quality['avg_accuracy']:.1%}")
    print(f"  Tasks w/ Consensus: {quality['tasks_with_consensus']}")
    print(f"  Labels Processed:  {quality['total_labels_processed']}")

    kappa_dist = quality['kappa_distribution']
    print(f"\n  Quality Distribution:")
    print(f"    Excellent (>0.75): {kappa_dist['excellent (>0.75)']}")
    print(f"    Good (0.60-0.75):  {kappa_dist['good (0.60-0.75)']}")
    print(f"    Fair (0.40-0.60):  {kappa_dist['fair (0.40-0.60)']}")
    print(f"    Poor (<0.40):      {kappa_dist['poor (<0.40)']}")

    print("\n## Marketplace")
    marketplace = summary['marketplace']
    print(f"  Tasks Created:   {marketplace['total_tasks_created']}")
    print(f"  Tasks Assigned:  {marketplace['total_tasks_assigned']}")
    print(f"  Tasks Completed: {marketplace['total_tasks_completed']}")
    print(f"  Avg Price/Label: ${marketplace['avg_price_per_label']:.3f}")

    # Save results
    print("\n[5/5] Saving results...")
    output_dir = Path(config['simulation'].get('output_dir', './results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save summary report
    report_path = run_dir / "summary_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary report: {report_path}")

    # Save config
    config_path = run_dir / "config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"  Configuration:  {config_path}")

    # Export sessions to CSV
    model.metrics_collector.export_to_csv(str(run_dir))
    print(f"  Sessions CSV:   {run_dir}/sessions.csv")

    # Generate visualizations
    try:
        model.metrics_collector.visualize_results(str(run_dir))
        print(f"  Visualizations: {run_dir}/simulation_results.png")
    except Exception as e:
        print(f"  Warning: Could not generate visualizations: {e}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {run_dir}")
    print("=" * 70)

    return model, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run labor-for-content-access simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/default.yaml)"
    )

    args = parser.parse_args()

    model, summary = main(config_path=args.config)
