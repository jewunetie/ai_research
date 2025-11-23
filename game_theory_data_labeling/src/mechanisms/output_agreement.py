"""Output Agreement mechanism - simple peer prediction."""

import numpy as np
from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.tasks.base import Task


class OutputAgreement(Mechanism):
    """Simple peer prediction: pay for agreement with peer.

    Algorithm:
    - Aggregate: Majority voting
    - Payment: Bonus for agreeing with randomly selected peer
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Configuration dict with optional keys:
                - agreement_payment: Payment when agreeing with peer (default 1.0)
                - disagreement_payment: Payment when disagreeing (default 0.0)
                - seed: Random seed for peer selection (default 42)
        """
        super().__init__(config or {})
        self.agreement_payment = self.config.get("agreement_payment", 1.0)
        self.disagreement_payment = self.config.get("disagreement_payment", 0.0)
        self.rng = np.random.default_rng(self.config.get("seed", 42))

    def requires_predictions(self) -> bool:
        """Output agreement doesn't need predictions."""
        return False

    def aggregate_and_pay(
        self, reports: list[Report], tasks: list[Task]
    ) -> MechanismResult:
        """Aggregate by majority, pay for agreement with random peer.

        Args:
            reports: All agent reports
            tasks: Task metadata (not used for output agreement)

        Returns:
            MechanismResult with aggregated labels and payments
        """
        # Group reports by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate by majority
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            votes = [r.report for r in task_reports]
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes) / 2 else 0

        # Payments: for each agent, compare with random peer on same tasks
        payments = {}
        for agent_id in set(r.agent_id for r in reports):
            agent_reports = [r for r in reports if r.agent_id == agent_id]
            total_payment = 0.0

            for report in agent_reports:
                # Find other agents who labeled same task
                peers = [
                    r
                    for r in reports
                    if r.task_id == report.task_id and r.agent_id != agent_id
                ]

                if peers:
                    # Select random peer
                    peer = peers[self.rng.integers(0, len(peers))]

                    # Pay for agreement
                    if report.report == peer.report:
                        total_payment += self.agreement_payment
                    else:
                        total_payment += self.disagreement_payment

            payments[agent_id] = total_payment

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=None,  # Output agreement doesn't estimate quality
        )
