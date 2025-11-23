"""Strategic agent implementation."""

from typing import TYPE_CHECKING
import numpy as np
from src.agents.base import Agent, AgentParams
from src.tasks.base import Task

if TYPE_CHECKING:
    from src.mechanisms.base import Mechanism


class StrategicAgent(Agent):
    """Agent that maximizes expected utility = payment - cost.

    This agent:
    - Observes the task with accuracy = ability
    - Strategically chooses report to maximize expected payment
    - Reasons about what other agents will report
    """

    def observe(self, task: Task) -> int:
        """Observe with accuracy = self.params.ability.

        Args:
            task: The task to observe

        Returns:
            signal: Observed label (0 or 1)
        """
        # With probability = ability, observe correctly
        if self.rng.random() < self.params.ability:
            return task.true_label
        else:
            return 1 - task.true_label

    def report(
        self,
        task: Task,
        signal: int,
        mechanism: "Mechanism",
        other_agents: list[Agent],
    ) -> int:
        """Report to maximize expected payment given beliefs about others.

        Args:
            task: The task to label
            signal: Agent's observation
            mechanism: The mechanism being used
            other_agents: Other agents in the game (for strategic reasoning)

        Returns:
            report: Best-response report (0 or 1)
        """
        # Compute expected payment for each possible report
        expected_payment = {}
        for possible_report in [0, 1]:
            # Estimate what others will report
            others_reports_dist = self._predict_others_reports(
                task, other_agents, mechanism
            )

            # Compute expected payment under this report
            expected_payment[possible_report] = self._compute_expected_payment(
                possible_report, others_reports_dist, mechanism
            )

        # Best-respond: report with highest expected payment
        if expected_payment[0] > expected_payment[1]:
            return 0
        elif expected_payment[1] > expected_payment[0]:
            return 1
        else:
            # Tie: report truthfully
            return signal

    def _predict_others_reports(
        self, task: Task, others: list[Agent], mechanism: "Mechanism"
    ) -> dict[int, float]:
        """Predict distribution of others' reports.

        Simplified model: Assume others report according to prior.
        More sophisticated: Model each agent type's strategy.

        Args:
            task: The task
            others: Other agents
            mechanism: The mechanism

        Returns:
            dict mapping label -> predicted probability
        """
        # Simple model: assume uniform (conservative assumption)
        # Future work: model agent types and mechanism
        return {0: 0.5, 1: 0.5}

    def _compute_expected_payment(
        self,
        my_report: int,
        others_dist: dict[int, float],
        mechanism: "Mechanism",
    ) -> float:
        """Compute expected payment given my report and beliefs about others.

        This is mechanism-specific and requires knowledge of payment rule.

        Args:
            my_report: My planned report (0 or 1)
            others_dist: Predicted distribution of others' reports
            mechanism: The mechanism being used

        Returns:
            expected_payment: Expected payment for this report
        """
        mechanism_name = mechanism.__class__.__name__

        if mechanism_name == "MajorityVoting":
            # Fixed payment regardless of report
            return mechanism.payment_per_task

        elif mechanism_name == "OutputAgreement":
            # Expected payment = P(others agree) * agreement_payment
            prob_agreement = others_dist.get(my_report, 0.5)
            return prob_agreement * mechanism.agreement_payment

        elif mechanism_name == "DawidSkene":
            # Simplified: assume quality-weighted payment
            # Strategic agent assumes they'll be classified as high-quality
            # This is optimistic but reasonable as first approximation
            return mechanism.payment_per_task * (
                1 - mechanism.quality_weight + mechanism.quality_weight * 0.8
            )

        elif mechanism_name == "RBTS":
            # For RBTS, payment depends on "surprisingly common" scoring
            # Simplified: assume moderate bonus for reporting
            return mechanism.base_payment + mechanism.bonus_scale * 0.5

        else:
            # Default: assume fixed payment
            return 1.0

    def predict_others(self, task: Task) -> dict[int, float]:
        """Predict distribution of others' reports (for RBTS).

        Args:
            task: The task

        Returns:
            dict mapping label -> predicted probability
        """
        # Strategic agent predicts uniform distribution
        # (Conservative, doesn't reveal private signal)
        return {0: 0.5, 1: 0.5}
