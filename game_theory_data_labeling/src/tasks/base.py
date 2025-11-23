"""Base task classes and task generation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    """A single labeling task with ground truth."""

    task_id: int
    true_label: int  # 0 or 1 for binary
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    prior_prob: float  # P(label=1) - prior probability

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, label={self.true_label}, diff={self.difficulty:.2f})"


class TaskGenerator:
    """Generates synthetic labeling tasks with ground truth."""

    def __init__(
        self,
        num_tasks: int,
        label_prior: float = 0.5,
        difficulty: float = 0.5,
        random_seed: int = 42,
    ):
        """
        Args:
            num_tasks: Number of tasks to generate
            label_prior: P(label=1) - probability of positive class
            difficulty: Task difficulty (0.0 easy to 1.0 hard)
            random_seed: Random seed for reproducibility
        """
        self.num_tasks = num_tasks
        self.label_prior = label_prior
        self.difficulty = difficulty
        self.rng = np.random.default_rng(random_seed)

    def generate(self) -> list[Task]:
        """Generate tasks with ground truth labels.

        Returns:
            List of Task objects
        """
        tasks = []
        for i in range(self.num_tasks):
            # Sample true label from prior
            true_label = 1 if self.rng.random() < self.label_prior else 0

            task = Task(
                task_id=i,
                true_label=true_label,
                difficulty=self.difficulty,
                prior_prob=self.label_prior,
            )
            tasks.append(task)

        return tasks
