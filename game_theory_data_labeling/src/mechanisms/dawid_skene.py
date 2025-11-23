"""Dawid-Skene EM algorithm implementation."""

import numpy as np
from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.tasks.base import Task


class DawidSkene(Mechanism):
    """Dawid-Skene EM algorithm for label aggregation.

    Algorithm:
    - Uses EM to jointly infer true labels and agent confusion matrices
    - E-step: Update label posteriors given confusion matrices
    - M-step: Update confusion matrices given label posteriors
    - Payment: Fixed + quality-weighted bonus
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Configuration dict with optional keys:
                - max_iterations: Maximum EM iterations (default 100)
                - tolerance: Convergence tolerance (default 1e-4)
                - payment_per_task: Base payment per task (default 1.0)
                - quality_weight: Weight for quality bonus 0-1 (default 0.5)
        """
        super().__init__(config or {})
        self.max_iterations = self.config.get("max_iterations", 100)
        self.tolerance = self.config.get("tolerance", 1e-4)
        self.payment_per_task = self.config.get("payment_per_task", 1.0)
        self.quality_weight = self.config.get("quality_weight", 0.5)

    def requires_predictions(self) -> bool:
        """Dawid-Skene doesn't need predictions."""
        return False

    def aggregate_and_pay(
        self, reports: list[Report], tasks: list[Task]
    ) -> MechanismResult:
        """Run EM to infer labels and quality, pay based on quality.

        Args:
            reports: All agent reports
            tasks: Task metadata

        Returns:
            MechanismResult with aggregated labels, payments, and quality estimates
        """
        # Convert reports to matrix format
        # reports_matrix[i, j] = label that agent i gave to task j
        # (or -1 if agent i didn't label task j)
        reports_matrix, agent_ids, task_ids = self._to_matrix(reports)
        n_agents, n_tasks = reports_matrix.shape
        n_classes = 2  # Binary for now

        # Initialize label posteriors using majority voting (better than uniform)
        # p[j, k] = P(true_label_j = k)
        p = np.zeros((n_tasks, n_classes))
        for j in range(n_tasks):
            # Count votes for this task
            votes = reports_matrix[:, j]
            votes = votes[votes >= 0]  # Remove -1 (unlabeled)
            if len(votes) > 0:
                majority_label = 1 if np.sum(votes) > len(votes) / 2 else 0
                p[j, majority_label] = 0.9  # High confidence in majority
                p[j, 1 - majority_label] = 0.1
            else:
                p[j, :] = 1.0 / n_classes  # Uniform if no votes

        # confusion[i, k, l] = P(agent i reports l | true label is k)
        # Initialize with slight bias towards correct reporting
        confusion = np.ones((n_agents, n_classes, n_classes)) * 0.2
        for i in range(n_agents):
            for k in range(n_classes):
                confusion[i, k, k] = 0.6  # Diagonal higher (correct reporting more likely)

        # EM iterations
        for iteration in range(self.max_iterations):
            # E-step: Update p given current confusion matrices
            p_new = self._e_step(reports_matrix, confusion, n_classes)

            # M-step: Update confusion matrices given current p
            confusion_new = self._m_step(reports_matrix, p_new, n_classes)

            # Check convergence of both label posteriors and confusion matrices
            p_converged = np.max(np.abs(p_new - p)) < self.tolerance
            confusion_converged = np.max(np.abs(confusion_new - confusion)) < self.tolerance

            if p_converged and confusion_converged:
                break

            p = p_new
            confusion = confusion_new

        # Aggregate: predicted label = argmax of p
        aggregated_labels = {task_ids[j]: int(np.argmax(p[j])) for j in range(n_tasks)}

        # Compute agent qualities
        # Quality = accuracy on diagonal of confusion matrix (average over classes)
        agent_qualities = {
            agent_ids[i]: (confusion[i, 0, 0] + confusion[i, 1, 1]) / 2
            for i in range(n_agents)
        }

        # Payment: fixed + bonus proportional to quality
        num_tasks_per_agent = np.sum(reports_matrix >= 0, axis=1)
        payments = {
            agent_ids[i]: self.payment_per_task
            * num_tasks_per_agent[i]
            * (
                1
                - self.quality_weight
                + self.quality_weight * agent_qualities[agent_ids[i]]
            )
            for i in range(n_agents)
        }

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=agent_qualities,
        )

    def _e_step(
        self, reports_matrix: np.ndarray, confusion: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """E-step: Update label posteriors.

        Args:
            reports_matrix: shape (n_agents, n_tasks)
            confusion: shape (n_agents, n_classes, n_classes)
            n_classes: Number of classes (2 for binary)

        Returns:
            p: shape (n_tasks, n_classes) - updated label posteriors
        """
        n_agents, n_tasks = reports_matrix.shape
        p = np.ones((n_tasks, n_classes)) / n_classes

        for j in range(n_tasks):
            for k in range(n_classes):
                # P(true_label_j = k | reports) ∝
                # P(true_label_j = k) * ∏_i P(report_ij | true_label_j = k)
                likelihood = 1.0
                for i in range(n_agents):
                    if reports_matrix[i, j] >= 0:  # If agent i labeled task j
                        l = int(reports_matrix[i, j])
                        likelihood *= confusion[i, k, l]
                p[j, k] = likelihood

            # Normalize
            if np.sum(p[j]) > 0:
                p[j] /= np.sum(p[j])
            else:
                # If all likelihoods are 0 (shouldn't happen), use uniform
                p[j] = np.ones(n_classes) / n_classes

        return p

    def _m_step(
        self, reports_matrix: np.ndarray, p: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """M-step: Update confusion matrices.

        Args:
            reports_matrix: shape (n_agents, n_tasks)
            p: shape (n_tasks, n_classes) - label posteriors
            n_classes: Number of classes (2 for binary)

        Returns:
            confusion: shape (n_agents, n_classes, n_classes) - updated confusion matrices
        """
        n_agents, n_tasks = reports_matrix.shape
        confusion = np.zeros((n_agents, n_classes, n_classes))

        for i in range(n_agents):
            for k in range(n_classes):
                for l in range(n_classes):
                    # confusion[i, k, l] =
                    # ∑_j p[j,k] * I(report_ij = l) / ∑_j p[j,k]
                    numerator = 0
                    denominator = 0
                    for j in range(n_tasks):
                        if reports_matrix[i, j] >= 0:
                            denominator += p[j, k]
                            if int(reports_matrix[i, j]) == l:
                                numerator += p[j, k]

                    if denominator > 1e-10:
                        confusion[i, k, l] = numerator / denominator
                    else:
                        # If denominator is 0, use uniform
                        confusion[i, k, l] = 1.0 / n_classes

        return confusion
