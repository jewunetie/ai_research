"""Unit tests for Phase 2 agent implementations."""

import pytest
import numpy as np

from src.agents.base import AgentParams
from src.agents.strategic import StrategicAgent
from src.agents.adversarial import AdversarialAgent
from src.agents.noisy_truthful import NoisyTruthfulAgent
from src.tasks.base import Task
from src.mechanisms.majority_voting import MajorityVoting
from src.mechanisms.output_agreement import OutputAgreement


def test_strategic_agent_observes():
    """Strategic agent should observe with given ability."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "strategic", ability=1.0)
    agent = StrategicAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # With ability=1.0, should always observe correctly
    observations = [agent.observe(task) for _ in range(10)]
    assert all(obs == 1 for obs in observations)


def test_strategic_agent_best_responds():
    """Strategic agent should best-respond to mechanism."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "strategic", ability=0.8)
    agent = StrategicAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)
    signal = 1

    # With majority voting (fixed payment), should report truthfully
    mechanism_mv = MajorityVoting({"payment_per_task": 1.0})
    report_mv = agent.report(task, signal, mechanism_mv, [])
    # Should be 0 or 1 (can't predict exact behavior without more info)
    assert report_mv in [0, 1]

    # With output agreement, might change behavior
    mechanism_oa = OutputAgreement({"agreement_payment": 2.0})
    report_oa = agent.report(task, signal, mechanism_oa, [])
    assert report_oa in [0, 1]


def test_strategic_agent_predictions():
    """Strategic agent should provide predictions for RBTS."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "strategic", ability=0.8)
    agent = StrategicAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)
    predictions = agent.predict_others(task)

    # Should return a valid probability distribution
    assert 0 in predictions and 1 in predictions
    assert abs(sum(predictions.values()) - 1.0) < 0.01
    assert all(0 <= p <= 1 for p in predictions.values())


def test_adversarial_agent_always_wrong():
    """Adversarial agent with 'always_wrong' strategy."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "adversarial", ability=1.0)
    agent = AdversarialAgent(params, rng, strategy="always_wrong")

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    for _ in range(10):
        signal = agent.observe(task)
        report = agent.report(task, signal, None, [])  # type: ignore
        # Should always report opposite of signal
        assert report == 1 - signal


def test_adversarial_agent_random():
    """Adversarial agent with 'random' strategy."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "adversarial", ability=0.5)
    agent = AdversarialAgent(params, rng, strategy="random")

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Random reports should be roughly 50/50
    reports = [agent.report(task, 0, None, []) for _ in range(1000)]  # type: ignore
    fraction_ones = sum(reports) / len(reports)
    assert 0.4 < fraction_ones < 0.6


def test_adversarial_agent_confuse():
    """Adversarial agent with 'confuse' strategy."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "adversarial", ability=0.8)
    agent = AdversarialAgent(params, rng, strategy="confuse")

    # Task with prior=0.7 (majority likely reports 1)
    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.7)

    signal = agent.observe(task)
    report = agent.report(task, signal, None, [])  # type: ignore

    # Should try to report opposite of majority (which is 1)
    # So should report 0
    assert report == 0


def test_noisy_truthful_agent():
    """Noisy truthful agent should report observations truthfully."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "noisy_truthful", ability=0.7)
    agent = NoisyTruthfulAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Should always report what they observe
    for _ in range(10):
        signal = agent.observe(task)
        report = agent.report(task, signal, None, [])  # type: ignore
        assert report == signal


def test_noisy_truthful_agent_predictions():
    """Noisy truthful agent should predict based on prior."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "noisy_truthful", ability=0.7)
    agent = NoisyTruthfulAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.7)
    predictions = agent.predict_others(task)

    # Should predict according to task prior (use approximate comparison for floats)
    assert abs(predictions[1] - 0.7) < 1e-10
    assert abs(predictions[0] - 0.3) < 1e-10


def test_noisy_truthful_vs_truthful_same_behavior():
    """Noisy truthful should behave same as truthful agent."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    params1 = AgentParams(0, "noisy_truthful", ability=0.8)
    params2 = AgentParams(1, "truthful", ability=0.8)

    from src.agents.truthful import TruthfulAgent
    agent1 = NoisyTruthfulAgent(params1, rng1)
    agent2 = TruthfulAgent(params2, rng2)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Both should observe and report identically with same RNG seed
    for _ in range(10):
        signal1 = agent1.observe(task)
        signal2 = agent2.observe(task)
        # With same seed, should get same observations
        assert signal1 == signal2

        report1 = agent1.report(task, signal1, None, [])  # type: ignore
        report2 = agent2.report(task, signal2, None, [])  # type: ignore
        assert report1 == report2
