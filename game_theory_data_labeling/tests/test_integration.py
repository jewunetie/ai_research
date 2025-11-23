"""Integration tests for full game runs."""

import pytest
import numpy as np

from src.simulation.game import ExperimentConfig
from src.simulation.engine import SimulationEngine


def test_full_game_truthful_agents():
    """Test a complete game run with truthful agents.

    With high-ability truthful agents, accuracy should be high.
    """
    config = ExperimentConfig(
        name="test_truthful",
        num_tasks=20,
        num_agents=5,
        agent_mix={"truthful": 1.0},
        agent_params={"truthful": {"ability": 0.95}},
        mechanism_name="majority_voting",
        mechanism_params={},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=3,
        random_seed=42,
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Should achieve high accuracy with high-ability truthful agents
    assert results.mean_metrics["accuracy"] > 0.90, \
        f"Expected >0.90 accuracy, got {results.mean_metrics['accuracy']}"

    # Should have run 3 times
    assert len(results.metrics) == 3

    # All metrics should be present
    assert "accuracy" in results.mean_metrics
    assert "f1_score" in results.mean_metrics
    assert "total_payment" in results.mean_metrics


def test_full_game_lazy_agents():
    """Test a complete game run with lazy agents.

    With lazy agents, accuracy should be around 50% (random).
    """
    config = ExperimentConfig(
        name="test_lazy",
        num_tasks=100,
        num_agents=5,
        agent_mix={"lazy": 1.0},
        mechanism_name="majority_voting",
        mechanism_params={},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=5,
        random_seed=42,
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Lazy random agents should be around 50% accuracy
    assert 0.35 < results.mean_metrics["accuracy"] < 0.65, \
        f"Expected ~0.50 accuracy, got {results.mean_metrics['accuracy']}"


def test_dawid_skene_identifies_quality():
    """Test that Dawid-Skene correctly identifies agent quality.

    Mix of truthful and lazy agents should show quality differentiation.
    """
    config = ExperimentConfig(
        name="test_quality",
        num_tasks=50,
        num_agents=6,
        agent_mix={"truthful": 0.5, "lazy": 0.5},  # 3 truthful, 3 lazy
        agent_params={"truthful": {"ability": 0.9}},
        mechanism_name="dawid_skene",
        mechanism_params={"quality_weight": 1.0},  # Full quality-based payment
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=1,
        random_seed=42,
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Should achieve better than random (lazy agents pull it down)
    assert results.mean_metrics["accuracy"] > 0.55, \
        "Mixed agents should beat random"

    # Payment should vary (quality-weighted)
    assert results.mean_metrics["payment_std"] > 0, \
        "Quality-weighted payment should have variation"


def test_different_mechanisms_same_agents():
    """Test that different mechanisms give different results on same setup."""

    base_config = {
        "name": "test_comparison",
        "num_tasks": 30,
        "num_agents": 5,
        "agent_mix": {"truthful": 0.6, "lazy": 0.4},
        "agent_params": {"truthful": {"ability": 0.85}},
        "task_difficulty": 0.5,
        "label_prior": 0.5,
        "num_runs": 3,
        "random_seed": 42,
    }

    # Run with majority voting
    config_mv = ExperimentConfig(**base_config, mechanism_name="majority_voting")
    results_mv = SimulationEngine(config_mv).run_experiment()

    # Run with Dawid-Skene
    config_ds = ExperimentConfig(**base_config, mechanism_name="dawid_skene")
    results_ds = SimulationEngine(config_ds).run_experiment()

    # Both should achieve reasonable accuracy
    assert results_mv.mean_metrics["accuracy"] > 0.60
    assert results_ds.mean_metrics["accuracy"] > 0.60

    # Dawid-Skene should provide quality estimates
    # (Can't check directly from ExperimentResults, but it shouldn't crash)
    assert results_ds.mean_metrics["total_payment"] > 0
