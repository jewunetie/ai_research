"""Lazy agent implementation."""

from typing import Literal
from src.agents.base import Agent, AgentParams
from src.tasks.base import Task
from src.mechanisms.base import Mechanism
import numpy as np


class LazyAgent(Agent):
    """Agent that minimizes effort.

    This agent:
    - Doesn't actually observe the task carefully
    - Reports randomly or according to prior
    - Minimizes effort to save cost
    """

    def __init__(
        self,
        params: AgentParams,
        rng: np.random.Generator,
        strategy: Literal["random", "prior"] = "random",
    ):
        """
        Args:
            params: Agent parameters
            rng: Random number generator
            strategy: 'random' (uniform) or 'prior' (follow prior distribution)
        """
        super().__init__(params, rng)
        self.strategy = strategy

    def observe(self, task: Task) -> int:
        """Lazy agent doesn't actually observe - saves effort.

        Args:
            task: The task (not actually observed)

        Returns:
            signal: Dummy value (not used in report anyway)
        """
        # Return dummy value (not used in report anyway)
        return 0

    def report(
        self,
        task: Task,
        signal: int,
        mechanism: Mechanism,
        other_agents: list[Agent],
    ) -> int:
        """Report without looking at signal.

        Args:
            task: The task
            signal: Observation (ignored by lazy agent)
            mechanism: The mechanism (ignored)
            other_agents: Other agents (ignored)

        Returns:
            report: Random or prior-based report
        """
        if self.strategy == "random":
            # Report uniformly random
            return self.rng.integers(0, 2)
        elif self.strategy == "prior":
            # Report based on prior probability
            return 1 if self.rng.random() < task.prior_prob else 0
        else:
            raise ValueError(f"Unknown lazy strategy: {self.strategy}")

    def predict_others(self, task: Task) -> dict[int, float]:
        """Predict distribution of others' reports (for RBTS).

        Args:
            task: The task

        Returns:
            dict mapping label -> predicted probability
        """
        # Lazy agent doesn't know what others will do
        # Predicts uniform distribution
        return {0: 0.5, 1: 0.5}
