"""Base mechanism classes."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.tasks.base import Task


@dataclass
class Report:
    """A single agent's report."""

    agent_id: int
    task_id: int
    report: int
    prediction: dict[int, float] | None = None  # For BTS/RBTS


@dataclass
class MechanismResult:
    """Result of running a mechanism on a set of reports."""

    aggregated_labels: dict[int, int]  # task_id -> predicted label
    payments: dict[int, float]  # agent_id -> payment
    agent_qualities: dict[int, float] | None = None  # For Dawid-Skene


class Mechanism(ABC):
    """Base class for all mechanisms."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def aggregate_and_pay(
        self, reports: list[Report], tasks: list["Task"]
    ) -> MechanismResult:
        """Aggregate reports and compute payments.

        Args:
            reports: All agent reports
            tasks: Task metadata (for some mechanisms)

        Returns:
            MechanismResult with aggregated labels and payments
        """
        pass

    @abstractmethod
    def requires_predictions(self) -> bool:
        """Does this mechanism need agents to predict others' reports?"""
        pass

    def _group_by_task(self, reports: list[Report]) -> dict[int, list[Report]]:
        """Helper: Group reports by task_id.

        Returns:
            dict mapping task_id -> list of reports for that task
        """
        grouped = {}
        for report in reports:
            if report.task_id not in grouped:
                grouped[report.task_id] = []
            grouped[report.task_id].append(report)
        return grouped

    def _to_matrix(
        self, reports: list[Report]
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Helper: Convert reports to matrix format for EM algorithms.

        Returns:
            reports_matrix: shape (n_agents, n_tasks), value = label or -1 if not labeled
            agent_ids: list of agent IDs corresponding to rows
            task_ids: list of task IDs corresponding to columns
        """
        # Get unique agents and tasks
        agent_ids = sorted(set(r.agent_id for r in reports))
        task_ids = sorted(set(r.task_id for r in reports))

        # Create mapping
        agent_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        task_to_idx = {tid: i for i, tid in enumerate(task_ids)}

        # Initialize matrix with -1 (not labeled)
        matrix = np.full((len(agent_ids), len(task_ids)), -1, dtype=int)

        # Fill in reports
        for report in reports:
            i = agent_to_idx[report.agent_id]
            j = task_to_idx[report.task_id]
            matrix[i, j] = report.report

        return matrix, agent_ids, task_ids
