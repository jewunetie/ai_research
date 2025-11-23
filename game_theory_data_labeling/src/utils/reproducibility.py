"""Reproducibility utilities for experiment management."""

import json
import subprocess
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np


class SeedManager:
    """Manages random seeds for reproducibility.

    Ensures deterministic random number generation across the entire
    experimental pipeline by managing seed allocation.
    """

    def __init__(self, master_seed: int):
        """Initialize seed manager with master seed.

        Args:
            master_seed: Master random seed for entire experiment
        """
        self.master_seed = master_seed
        self.master_rng = np.random.default_rng(master_seed)

    def get_game_seed(self, run_id: int) -> int:
        """Get seed for a specific game run.

        Args:
            run_id: Game run identifier (0-indexed)

        Returns:
            Deterministic seed for this game run
        """
        return self.master_seed + run_id

    def get_agent_seed(self, agent_id: int, game_seed: int) -> int:
        """Get seed for a specific agent.

        Args:
            agent_id: Agent identifier
            game_seed: Seed for the current game

        Returns:
            Deterministic seed for this agent
        """
        return game_seed + 1000 * agent_id

    def get_task_seed(self, game_seed: int) -> int:
        """Get seed for task generation.

        Args:
            game_seed: Seed for the current game

        Returns:
            Deterministic seed for task generation
        """
        return game_seed + 10000


def get_git_commit_hash() -> str:
    """Get current git commit hash for provenance tracking.

    Returns:
        Git commit hash or 'unknown' if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def log_experiment(config: Any, results: Any, output_dir: str = "results/raw") -> Path:
    """Log all experiment details to JSON file.

    Args:
        config: ExperimentConfig object
        results: ExperimentResults object
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else config,
        "results": {
            "mean_metrics": results.mean_metrics,
            "std_metrics": results.std_metrics,
            "confidence_intervals": {
                k: list(v) for k, v in results.confidence_intervals.items()
            }
        }
    }

    # Generate filename with timestamp
    filename = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = output_path / filename

    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    return file_path


def save_results_summary(results_dict: dict[str, Any], output_file: str = "results/summary.json"):
    """Save summary of multiple experiments to JSON.

    Args:
        results_dict: Dictionary mapping experiment name to results
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "experiments": results_dict
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return output_path


def load_experiment_results(file_path: str) -> dict:
    """Load experiment results from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with experiment data
    """
    with open(file_path, 'r') as f:
        return json.load(f)
