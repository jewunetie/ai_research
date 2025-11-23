"""Unit tests for agent implementations."""

import pytest
import numpy as np

from src.agents.base import AgentParams
from src.agents.truthful import TruthfulAgent
from src.agents.lazy import LazyAgent
from src.tasks.base import Task


def test_truthful_agent_high_ability():
    """Truthful agent with ability=1.0 should always observe correctly."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "truthful", ability=1.0, effort_cost=0, risk_aversion=0)
    agent = TruthfulAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Run 100 times, should always observe correctly
    observations = [agent.observe(task) for _ in range(100)]
    assert all(obs == 1 for obs in observations), "High ability agent should always observe correctly"


def test_truthful_agent_reports_observation():
    """Truthful agent should always report their observation."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "truthful", ability=0.8, effort_cost=0, risk_aversion=0)
    agent = TruthfulAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Test that report equals observation
    for _ in range(10):
        signal = agent.observe(task)
        report = agent.report(task, signal, None, [])  # type: ignore
        assert report == signal, "Truthful agent should report their observation"


def test_lazy_agent_random():
    """Lazy agent with random strategy should report roughly 50/50."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "lazy", ability=0.5, effort_cost=0, risk_aversion=0)
    agent = LazyAgent(params, rng, strategy="random")

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Run many times, should be roughly 50/50
    reports = [agent.report(task, 0, None, []) for _ in range(1000)]  # type: ignore
    fraction_ones = sum(reports) / len(reports)
    assert 0.4 < fraction_ones < 0.6, f"Lazy random should be ~0.5, got {fraction_ones}"


def test_lazy_agent_prior():
    """Lazy agent with prior strategy should follow prior distribution."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "lazy", ability=0.5, effort_cost=0, risk_aversion=0)
    agent = LazyAgent(params, rng, strategy="prior")

    # Task with prior = 0.7
    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.7)

    # Run many times, should be roughly 70% ones
    reports = [agent.report(task, 0, None, []) for _ in range(1000)]  # type: ignore
    fraction_ones = sum(reports) / len(reports)
    assert 0.6 < fraction_ones < 0.8, f"Lazy prior should be ~0.7, got {fraction_ones}"
