"""Majority voting mechanism implementation."""

from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.tasks.base import Task


class MajorityVoting(Mechanism):
    """Simple majority voting with fixed payment.

    Algorithm:
    - Aggregate: Select label with >50% of votes
    - Payment: Fixed payment per task labeled
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Configuration dict with optional keys:
                - payment_per_task: Fixed payment per task (default 1.0)
        """
        super().__init__(config or {})
        self.payment_per_task = self.config.get("payment_per_task", 1.0)

    def requires_predictions(self) -> bool:
        """Majority voting doesn't need predictions."""
        return False

    def aggregate_and_pay(
        self, reports: list[Report], tasks: list[Task]
    ) -> MechanismResult:
        """Aggregate by majority, pay fixed amount.

        Args:
            reports: All agent reports
            tasks: Task metadata (not used for majority voting)

        Returns:
            MechanismResult with aggregated labels and payments
        """
        # Group reports by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate each task by majority vote
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            # Count votes
            votes = [r.report for r in task_reports]
            # Majority: if sum > half, then label=1, else label=0
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes) / 2 else 0

        # Pay everyone equally based on number of tasks they labeled
        unique_agents = set(r.agent_id for r in reports)
        payments = {
            agent_id: self.payment_per_task
            * len(set(r.task_id for r in reports if r.agent_id == agent_id))
            for agent_id in unique_agents
        }

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=None,  # Majority voting doesn't estimate quality
        )
