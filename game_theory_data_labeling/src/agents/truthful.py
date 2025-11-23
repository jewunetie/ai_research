"""Truthful agent implementation."""

from src.agents.base import Agent, AgentParams
from src.tasks.base import Task
from src.mechanisms.base import Mechanism
import numpy as np


class TruthfulAgent(Agent):
    """Agent that always reports their observation.

    This agent:
    - Observes the task with accuracy = ability
    - Always reports their observation truthfully
    - Does not strategize about payments
    """

    def observe(self, task: Task) -> int:
        """Observe with accuracy = self.params.ability.

        Args:
            task: The task to observe

        Returns:
            signal: Observed label (0 or 1)
        """
        # With probability = ability, observe correctly
        # Otherwise, observe incorrectly
        if self.rng.random() < self.params.ability:
            return task.true_label
        else:
            return 1 - task.true_label

    def report(
        self,
        task: Task,
        signal: int,
        mechanism: Mechanism,
        other_agents: list[Agent],
    ) -> int:
        """Truthful agent always reports their signal.

        Args:
            task: The task to label
            signal: Agent's observation
            mechanism: The mechanism (ignored by truthful agent)
            other_agents: Other agents (ignored by truthful agent)

        Returns:
            report: The signal (truthful reporting)
        """
        return signal
