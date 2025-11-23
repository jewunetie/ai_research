"""Unit tests for mechanism implementations."""

import pytest
import numpy as np

from src.mechanisms.base import Report
from src.mechanisms.majority_voting import MajorityVoting
from src.mechanisms.dawid_skene import DawidSkene
from src.tasks.base import Task


def test_majority_voting_simple():
    """Majority voting should select majority label."""
    mechanism = MajorityVoting({"payment_per_task": 1.0})

    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 1),  # Agent 1, Task 0, Label 1
        Report(2, 0, 0),  # Agent 2, Task 0, Label 0
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    assert result.aggregated_labels[0] == 1, "Majority is 1 (2 out of 3)"
    assert all(p == 1.0 for p in result.payments.values()), "All agents paid equally"
    assert len(result.payments) == 3, "3 agents should be paid"


def test_majority_voting_tie():
    """Majority voting with tie should pick 0 (sum <= len/2)."""
    mechanism = MajorityVoting({"payment_per_task": 1.0})

    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 0),  # Agent 1, Task 0, Label 0
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # With tie (1 vote each), sum(votes) = 1, len(votes)/2 = 1.0, so 1 is not > 1.0
    assert result.aggregated_labels[0] == 0, "Tie should resolve to 0"


def test_majority_voting_multiple_tasks():
    """Majority voting with multiple tasks."""
    mechanism = MajorityVoting({"payment_per_task": 1.0})

    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 1),  # Agent 1, Task 0, Label 1
        Report(0, 1, 0),  # Agent 0, Task 1, Label 0
        Report(1, 1, 0),  # Agent 1, Task 1, Label 0
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    assert result.aggregated_labels[0] == 1, "Task 0 majority is 1"
    assert result.aggregated_labels[1] == 0, "Task 1 majority is 0"
    # Each agent labeled 2 tasks
    assert result.payments[0] == 2.0, "Agent 0 labeled 2 tasks"
    assert result.payments[1] == 2.0, "Agent 1 labeled 2 tasks"


def test_dawid_skene_convergence():
    """Dawid-Skene should converge and aggregate correctly."""
    mechanism = DawidSkene({
        "max_iterations": 100,
        "tolerance": 1e-4,
        "payment_per_task": 1.0,
        "quality_weight": 0.5,
    })

    # Synthetic data: 3 agents label 5 tasks
    # Agent 0: always correct (high quality)
    # Agent 1: always correct (high quality)
    # Agent 2: random (low quality)
    # True labels: [1, 0, 1, 0, 1]
    reports = [
        # Task 0 (true=1): agents report [1, 1, 0]
        Report(0, 0, 1),
        Report(1, 0, 1),
        Report(2, 0, 0),
        # Task 1 (true=0): agents report [0, 0, 1]
        Report(0, 1, 0),
        Report(1, 1, 0),
        Report(2, 1, 1),
        # Task 2 (true=1): agents report [1, 1, 1]
        Report(0, 2, 1),
        Report(1, 2, 1),
        Report(2, 2, 1),
        # Task 3 (true=0): agents report [0, 0, 0]
        Report(0, 3, 0),
        Report(1, 3, 0),
        Report(2, 3, 0),
        # Task 4 (true=1): agents report [1, 1, 0]
        Report(0, 4, 1),
        Report(1, 4, 1),
        Report(2, 4, 0),
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Should get majority for all tasks
    assert result.aggregated_labels[0] == 1
    assert result.aggregated_labels[1] == 0
    assert result.aggregated_labels[2] == 1
    assert result.aggregated_labels[3] == 0
    assert result.aggregated_labels[4] == 1

    # Should identify agent 0 and 1 as higher quality than agent 2
    assert result.agent_qualities is not None
    assert result.agent_qualities[0] > result.agent_qualities[2]
    assert result.agent_qualities[1] > result.agent_qualities[2]

    # Higher quality agents should get higher payment (with quality_weight > 0)
    assert result.payments[0] > result.payments[2]
    assert result.payments[1] > result.payments[2]


def test_dawid_skene_perfect_agents():
    """Dawid-Skene with perfect agents should identify quality=1.0."""
    mechanism = DawidSkene({
        "max_iterations": 100,
        "tolerance": 1e-4,
    })

    # 2 perfect agents, 3 tasks, all agree correctly
    # True labels: [1, 0, 1]
    reports = [
        Report(0, 0, 1),
        Report(1, 0, 1),
        Report(0, 1, 0),
        Report(1, 1, 0),
        Report(0, 2, 1),
        Report(1, 2, 1),
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Both agents should have high quality (close to 1.0)
    assert result.agent_qualities is not None
    assert result.agent_qualities[0] > 0.95
    assert result.agent_qualities[1] > 0.95
