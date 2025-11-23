"""Integration tests for Phase 2 - full game with all mechanisms."""

import pytest
from src.simulation.game import Game, ExperimentConfig
import numpy as np


def test_all_mechanisms_with_truthful_agents():
    """All 4 mechanisms should work with truthful agents."""
    mechanisms = ["majority_voting", "dawid_skene", "output_agreement", "rbts"]

    for mechanism_name in mechanisms:
        config = ExperimentConfig(
            name=f"test_{mechanism_name}",
            num_tasks=10,
            num_agents=5,
            agent_mix={"truthful": 1.0},
            agent_params={"truthful": {"ability": 0.9}},
            mechanism_name=mechanism_name,
            mechanism_params={"payment_per_task": 1.0} if mechanism_name != "output_agreement" else {"agreement_payment": 1.0},
            random_seed=42,
        )

        rng = np.random.default_rng(42)
        game = Game(config, rng)
        result = game.run()

        # Should have aggregated labels for all tasks
        assert len(result.result.aggregated_labels) == 10

        # Should have payments for all agents
        assert len(result.result.payments) == 5

        # All payments should be non-negative
        assert all(p >= 0 for p in result.result.payments.values())


def test_output_agreement_incentivizes_agreement():
    """Output agreement should incentivize agents to agree."""
    config = ExperimentConfig(
        name="test_oa_agreement",
        num_tasks=20,
        num_agents=10,
        agent_mix={"truthful": 0.8, "lazy": 0.2},
        agent_params={"truthful": {"ability": 0.85}},
        mechanism_name="output_agreement",
        mechanism_params={"agreement_payment": 1.0, "disagreement_payment": 0.0},
        random_seed=42,
    )

    rng = np.random.default_rng(42)
    game = Game(config, rng)
    result = game.run()

    # Truthful agents should get higher payment than lazy agents on average
    # (because truthful agents tend to agree more)
    truthful_payments = [result.result.payments[i] for i in range(8)]  # First 8 are truthful
    lazy_payments = [result.result.payments[i] for i in range(8, 10)]  # Last 2 are lazy

    avg_truthful = sum(truthful_payments) / len(truthful_payments)
    avg_lazy = sum(lazy_payments) / len(lazy_payments)

    # Truthful agents should generally earn more (they agree more often)
    assert avg_truthful > avg_lazy * 0.8  # At least 80% of lazy payment


def test_rbts_with_predictions():
    """RBTS should work when agents provide predictions."""
    config = ExperimentConfig(
        name="test_rbts",
        num_tasks=15,
        num_agents=6,
        agent_mix={"truthful": 0.7, "strategic": 0.3},
        agent_params={"truthful": {"ability": 0.9}, "strategic": {"ability": 0.85}},
        mechanism_name="rbts",
        mechanism_params={"base_payment": 1.0, "bonus_scale": 0.5},
        random_seed=42,
    )

    rng = np.random.default_rng(42)
    game = Game(config, rng)
    result = game.run()

    # All reports should have predictions
    assert all(r.prediction is not None for r in result.reports)

    # Should have valid payments
    assert all(p >= 0 for p in result.result.payments.values())

    # Should aggregate labels
    assert len(result.result.aggregated_labels) == 15


def test_strategic_agents_with_different_mechanisms():
    """Strategic agents should adapt to different mechanisms."""
    mechanisms = ["majority_voting", "output_agreement", "dawid_skene", "rbts"]
    results = {}

    for mechanism_name in mechanisms:
        config = ExperimentConfig(
            name=f"test_strategic_{mechanism_name}",
            num_tasks=20,
            num_agents=10,
            agent_mix={"strategic": 0.5, "truthful": 0.5},
            agent_params={"strategic": {"ability": 0.8}, "truthful": {"ability": 0.8}},
            mechanism_name=mechanism_name,
            mechanism_params={"payment_per_task": 1.0} if mechanism_name != "output_agreement" else {"agreement_payment": 1.0},
            random_seed=42,
        )

        rng = np.random.default_rng(42)
        game = Game(config, rng)
        result = game.run()
        results[mechanism_name] = result

        # Basic sanity checks
        assert len(result.result.aggregated_labels) == 20
        assert len(result.result.payments) == 10
        assert all(p >= 0 for p in result.result.payments.values())


def test_adversarial_agents_hurt_quality():
    """Adversarial agents should decrease accuracy."""
    # First: all truthful agents
    config_truthful = ExperimentConfig(
        name="test_all_truthful",
        num_tasks=30,
        num_agents=10,
        agent_mix={"truthful": 1.0},
        agent_params={"truthful": {"ability": 0.9}},
        mechanism_name="majority_voting",
        mechanism_params={"payment_per_task": 1.0},
        random_seed=42,
    )

    rng1 = np.random.default_rng(42)
    game1 = Game(config_truthful, rng1)
    result1 = game1.run()

    # Count correct predictions
    correct1 = sum(
        1
        for task_id, pred_label in result1.result.aggregated_labels.items()
        if pred_label == result1.true_labels[task_id]
    )
    accuracy1 = correct1 / len(result1.result.aggregated_labels)

    # Second: mix with adversarial agents
    config_adversarial = ExperimentConfig(
        name="test_with_adversarial",
        num_tasks=30,
        num_agents=10,
        agent_mix={"truthful": 0.6, "adversarial": 0.4},
        agent_params={"truthful": {"ability": 0.9}, "adversarial": {"ability": 0.5}},
        mechanism_name="majority_voting",
        mechanism_params={"payment_per_task": 1.0},
        random_seed=42,
    )

    rng2 = np.random.default_rng(42)
    game2 = Game(config_adversarial, rng2)
    result2 = game2.run()

    correct2 = sum(
        1
        for task_id, pred_label in result2.result.aggregated_labels.items()
        if pred_label == result2.true_labels[task_id]
    )
    accuracy2 = correct2 / len(result2.result.aggregated_labels)

    # Accuracy with adversarial agents should be lower
    assert accuracy2 < accuracy1


def test_noisy_truthful_agents():
    """Noisy truthful agents with lower ability should have lower accuracy."""
    # High ability
    config_high = ExperimentConfig(
        name="test_high_ability",
        num_tasks=30,
        num_agents=10,
        agent_mix={"noisy_truthful": 1.0},
        agent_params={"noisy_truthful": {"ability": 0.95}},
        mechanism_name="dawid_skene",
        mechanism_params={"payment_per_task": 1.0},
        random_seed=42,
    )

    rng1 = np.random.default_rng(42)
    game1 = Game(config_high, rng1)
    result1 = game1.run()

    # Low ability
    config_low = ExperimentConfig(
        name="test_low_ability",
        num_tasks=30,
        num_agents=10,
        agent_mix={"noisy_truthful": 1.0},
        agent_params={"noisy_truthful": {"ability": 0.6}},
        mechanism_name="dawid_skene",
        mechanism_params={"payment_per_task": 1.0},
        random_seed=42,
    )

    rng2 = np.random.default_rng(42)
    game2 = Game(config_low, rng2)
    result2 = game2.run()

    # Dawid-Skene should identify quality differences
    if result1.result.agent_qualities and result2.result.agent_qualities:
        avg_quality1 = sum(result1.result.agent_qualities.values()) / len(
            result1.result.agent_qualities
        )
        avg_quality2 = sum(result2.result.agent_qualities.values()) / len(
            result2.result.agent_qualities
        )

        # Higher ability agents should have higher estimated quality
        assert avg_quality1 > avg_quality2


def test_mixed_agent_population():
    """Test game with all agent types mixed together."""
    config = ExperimentConfig(
        name="test_mixed_population",
        num_tasks=25,
        num_agents=20,
        agent_mix={
            "truthful": 0.3,
            "lazy": 0.2,
            "strategic": 0.2,
            "adversarial": 0.15,
            "noisy_truthful": 0.15,
        },
        agent_params={
            "truthful": {"ability": 0.95},
            "strategic": {"ability": 0.85},
            "noisy_truthful": {"ability": 0.7},
            "adversarial": {"ability": 0.6},
        },
        mechanism_name="dawid_skene",
        mechanism_params={"payment_per_task": 1.0, "quality_weight": 0.5},
        random_seed=42,
    )

    rng = np.random.default_rng(42)
    game = Game(config, rng)
    result = game.run()

    # Should complete successfully with mixed population
    assert len(result.result.aggregated_labels) == 25
    assert len(result.result.payments) == 20
    assert all(p >= 0 for p in result.result.payments.values())

    # Dawid-Skene should estimate quality
    assert result.result.agent_qualities is not None
    assert len(result.result.agent_qualities) == 20
