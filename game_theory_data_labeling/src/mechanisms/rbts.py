"""Robust Bayesian Truth Serum (RBTS) mechanism implementation."""

import numpy as np
from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.tasks.base import Task


class RBTS(Mechanism):
    """Robust Bayesian Truth Serum for small populations.

    Algorithm:
    - Requires agents to predict others' reports
    - Rewards "surprisingly common" answers
    - Score = log(actual frequency / predicted frequency)
    - Theoretically incentive-compatible

    Based on Witkowski & Parkes (2012): "A Robust Bayesian Truth Serum for Small Populations"
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Configuration dict with optional keys:
                - base_payment: Base payment per task (default 1.0)
                - bonus_scale: Scale factor for RBTS bonus (default 1.0)
        """
        super().__init__(config or {})
        self.base_payment = self.config.get("base_payment", 1.0)
        self.bonus_scale = self.config.get("bonus_scale", 1.0)

    def requires_predictions(self) -> bool:
        """RBTS requires predictions of others' reports."""
        return True

    def aggregate_and_pay(
        self, reports: list[Report], tasks: list[Task]
    ) -> MechanismResult:
        """RBTS scoring: reward surprisingly common answers.

        Args:
            reports: All agent reports (must include predictions)
            tasks: Task metadata

        Returns:
            MechanismResult with aggregated labels and payments

        Raises:
            ValueError: If any report is missing predictions
        """
        # Group by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate by majority
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            votes = [r.report for r in task_reports]
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes) / 2 else 0

        # RBTS payments
        payments = {}
        MIN_PREDICTION = 0.01  # Minimum allowed prediction
        MIN_FREQUENCY = 1e-10  # Avoid log(0)

        for agent_id in set(r.agent_id for r in reports):
            agent_reports = [r for r in reports if r.agent_id == agent_id]
            total_payment = 0.0

            for report in agent_reports:
                # Validate prediction exists
                if report.prediction is None:
                    raise ValueError(
                        f"RBTS requires predictions, but agent {agent_id} didn't provide one"
                    )

                # Get all reports for this task
                task_reports = reports_by_task[report.task_id]
                n = len(task_reports)

                # Frequency of my report (among all agents)
                my_report_count = sum(
                    1 for r in task_reports if r.report == report.report
                )
                frequency = my_report_count / n if n > 0 else 0

                # My prediction for my own report
                my_prediction = report.prediction.get(report.report, 0.5)

                # RBTS score (simplified version)
                # Full version has more sophisticated scoring
                if my_prediction >= MIN_PREDICTION and frequency > MIN_FREQUENCY:
                    # Clip ratio to prevent extreme scores from gaming
                    ratio = np.clip(frequency / my_prediction, 0.01, 100.0)
                    score = np.log(max(ratio, MIN_FREQUENCY))
                else:
                    score = 0.0

                total_payment += self.base_payment + self.bonus_scale * score

            # Ensure non-negative payment
            payments[agent_id] = max(0.0, total_payment)

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=None,  # RBTS doesn't estimate quality
        )
