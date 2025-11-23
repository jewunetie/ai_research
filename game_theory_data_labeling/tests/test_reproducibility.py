"""Tests for reproducibility utilities."""

import pytest
import json
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass

from src.utils.reproducibility import (
    SeedManager,
    get_git_commit_hash,
    log_experiment,
    save_results_summary,
    load_experiment_results
)


def test_seed_manager_deterministic():
    """SeedManager should produce deterministic seeds."""
    manager1 = SeedManager(42)
    manager2 = SeedManager(42)

    # Same master seed should produce same game seeds
    assert manager1.get_game_seed(0) == manager2.get_game_seed(0)
    assert manager1.get_game_seed(10) == manager2.get_game_seed(10)

    # Same inputs should produce same agent seeds
    game_seed = manager1.get_game_seed(0)
    assert manager1.get_agent_seed(0, game_seed) == manager2.get_agent_seed(0, game_seed)
    assert manager1.get_agent_seed(5, game_seed) == manager2.get_agent_seed(5, game_seed)


def test_seed_manager_different_runs():
    """Different runs should get different seeds."""
    manager = SeedManager(42)

    seed0 = manager.get_game_seed(0)
    seed1 = manager.get_game_seed(1)
    seed2 = manager.get_game_seed(2)

    # All different
    assert seed0 != seed1
    assert seed1 != seed2
    assert seed0 != seed2


def test_seed_manager_different_agents():
    """Different agents should get different seeds."""
    manager = SeedManager(42)
    game_seed = manager.get_game_seed(0)

    agent_seed0 = manager.get_agent_seed(0, game_seed)
    agent_seed1 = manager.get_agent_seed(1, game_seed)
    agent_seed2 = manager.get_agent_seed(2, game_seed)

    # All different
    assert agent_seed0 != agent_seed1
    assert agent_seed1 != agent_seed2
    assert agent_seed0 != agent_seed2


def test_git_commit_hash():
    """git_commit_hash should return string."""
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    # Either valid hash or 'unknown'
    assert len(commit) > 0


def test_log_experiment():
    """log_experiment should save experiment data to JSON."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock config and results
        @dataclass
        class MockConfig:
            name: str
            num_tasks: int
            num_agents: int

        @dataclass
        class MockResults:
            mean_metrics: dict
            std_metrics: dict
            confidence_intervals: dict

        config = MockConfig(name="test_exp", num_tasks=10, num_agents=5)
        results = MockResults(
            mean_metrics={"accuracy": 0.95},
            std_metrics={"accuracy": 0.02},
            confidence_intervals={"accuracy": (0.93, 0.97)}
        )

        # Log experiment
        file_path = log_experiment(config, results, output_dir=tmpdir)

        # Verify file was created
        assert file_path.exists()
        assert file_path.suffix == '.json'

        # Load and verify contents
        with open(file_path, 'r') as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "git_commit" in data
        assert "config" in data
        assert data["config"]["name"] == "test_exp"
        assert data["results"]["mean_metrics"]["accuracy"] == 0.95


def test_save_results_summary():
    """save_results_summary should save multiple experiment results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "summary.json"

        results_dict = {
            "exp1": {"accuracy": 0.95, "f1": 0.94},
            "exp2": {"accuracy": 0.92, "f1": 0.91}
        }

        # Save summary
        path = save_results_summary(results_dict, output_file=str(output_file))

        # Verify file was created
        assert path.exists()

        # Load and verify
        data = load_experiment_results(str(path))
        assert "timestamp" in data
        assert "experiments" in data
        assert "exp1" in data["experiments"]
        assert data["experiments"]["exp1"]["accuracy"] == 0.95


def test_load_experiment_results():
    """load_experiment_results should load JSON correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test JSON file
        test_data = {
            "config": {"name": "test"},
            "results": {"accuracy": 0.95}
        }

        file_path = Path(tmpdir) / "test.json"
        with open(file_path, 'w') as f:
            json.dump(test_data, f)

        # Load
        loaded = load_experiment_results(str(file_path))

        assert loaded["config"]["name"] == "test"
        assert loaded["results"]["accuracy"] == 0.95
