"""Unit tests for Phase 2 mechanism implementations."""

import pytest
import numpy as np

from src.mechanisms.base import Report
from src.mechanisms.output_agreement import OutputAgreement
from src.mechanisms.rbts import RBTS
from src.tasks.base import Task


def test_output_agreement_all_agree():
    """Output agreement should pay when agents agree."""
    mechanism = OutputAgreement({"agreement_payment": 1.0, "disagreement_payment": 0.0, "seed": 42})

    # All agents agree on task 0
    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 1),  # Agent 1, Task 0, Label 1
        Report(2, 0, 1),  # Agent 2, Task 0, Label 1
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Should aggregate to 1
    assert result.aggregated_labels[0] == 1

    # Each agent should get paid for agreeing with peers
    # Each agent compared with 1 random peer per task
    assert result.payments[0] == 1.0  # Agreed with peer
    assert result.payments[1] == 1.0
    assert result.payments[2] == 1.0


def test_output_agreement_disagreement():
    """Output agreement should pay less when agents disagree."""
    mechanism = OutputAgreement({"agreement_payment": 1.0, "disagreement_payment": 0.0, "seed": 42})

    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 0),  # Agent 1, Task 0, Label 0
        Report(2, 0, 1),  # Agent 2, Task 0, Label 1
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Majority is 1
    assert result.aggregated_labels[0] == 1

    # Payments depend on agreement with random peer
    # Not all agents will get full payment
    total_payment = sum(result.payments.values())
    assert total_payment < 3.0  # Less than if all agreed


def test_rbts_requires_predictions():
    """RBTS should require predictions."""
    mechanism = RBTS({"base_payment": 1.0, "bonus_scale": 1.0})
    assert mechanism.requires_predictions() is True


def test_rbts_with_predictions():
    """RBTS should compute scores based on predictions."""
    mechanism = RBTS({"base_payment": 1.0, "bonus_scale": 1.0})

    # 3 agents, all report 1
    # Agent 0 predicts 50/50, Agent 1 predicts 90% will report 1, Agent 2 predicts 10% will report 1
    reports = [
        Report(0, 0, 1, prediction={0: 0.5, 1: 0.5}),  # Accurate prediction
        Report(1, 0, 1, prediction={0: 0.1, 1: 0.9}),  # Over-confident (predicted high, got high)
        Report(2, 0, 1, prediction={0: 0.9, 1: 0.1}),  # Under-confident (predicted low, got high)
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # All reported 1, so majority is 1
    assert result.aggregated_labels[0] == 1

    # RBTS scoring:
    # Agent 0: frequency=1.0, prediction=0.5, score=log(1.0/0.5)=log(2)≈0.69
    # Agent 1: frequency=1.0, prediction=0.9, score=log(1.0/0.9)≈0.11
    # Agent 2: frequency=1.0, prediction=0.1, score=log(1.0/0.1)=log(10)≈2.30

    # Agent 2 should get highest bonus (surprisingly common)
    assert result.payments[2] > result.payments[1]
    assert result.payments[2] > result.payments[0]

    # All payments should be positive
    assert all(p > 0 for p in result.payments.values())


def test_rbts_missing_prediction():
    """RBTS should raise error if prediction is missing."""
    mechanism = RBTS({"base_payment": 1.0, "bonus_scale": 1.0})

    reports = [
        Report(0, 0, 1, prediction=None),  # Missing prediction
        Report(1, 0, 1, prediction={0: 0.5, 1: 0.5}),
    ]

    with pytest.raises(ValueError, match="RBTS requires predictions"):
        mechanism.aggregate_and_pay(reports, [])


def test_rbts_numerical_safety():
    """RBTS should handle edge cases in predictions."""
    mechanism = RBTS({"base_payment": 1.0, "bonus_scale": 1.0})

    # Test with extreme predictions
    reports = [
        Report(0, 0, 1, prediction={0: 0.99, 1: 0.01}),  # Very low prediction
        Report(1, 0, 1, prediction={0: 0.01, 1: 0.99}),  # Very high prediction
        Report(2, 0, 1, prediction={0: 0.001, 1: 0.999}),  # Extreme prediction
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Should not crash and should produce non-negative payments
    assert all(p >= 0 for p in result.payments.values())
    assert result.aggregated_labels[0] == 1


def test_output_agreement_multiple_tasks():
    """Output agreement with multiple tasks."""
    mechanism = OutputAgreement({"agreement_payment": 1.0, "seed": 42})

    reports = [
        # Task 0: agents agree
        Report(0, 0, 1),
        Report(1, 0, 1),
        # Task 1: agents disagree
        Report(0, 1, 0),
        Report(1, 1, 1),
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    # Check aggregation
    assert result.aggregated_labels[0] == 1
    # Task 1 is a tie (sum=1, len/2=1), so should be 0
    assert result.aggregated_labels[1] == 0

    # Both agents labeled 2 tasks, but payments depend on agreement
    assert 0 in result.payments
    assert 1 in result.payments
