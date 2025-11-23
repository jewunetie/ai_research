"""Noisy truthful agent implementation."""

from typing import TYPE_CHECKING
import numpy as np
from src.agents.base import Agent, AgentParams
from src.tasks.base import Task

if TYPE_CHECKING:
    from src.mechanisms.base import Mechanism


class NoisyTruthfulAgent(Agent):
    """Agent that is truthful but has noisy observations.

    This agent:
    - Observes with noise (accuracy = ability)
    - Always reports their observation truthfully
    - Represents well-intentioned but less skilled annotators

    Note: This is essentially the same as TruthfulAgent, but
    conceptually separated for clarity in agent mix descriptions.
    """

    def observe(self, task: Task) -> int:
        """Observe with noise based on ability.

        Args:
            task: The task to observe

        Returns:
            signal: Noisy observation (0 or 1)
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
        """Truthfully report observation.

        Args:
            task: The task
            signal: Agent's noisy observation
            mechanism: The mechanism (ignored)
            other_agents: Other agents (ignored)

        Returns:
            report: The signal (truthful reporting)
        """
        return signal

    def predict_others(self, task: Task) -> dict[int, float]:
        """Predict distribution of others' reports (for RBTS).

        Args:
            task: The task

        Returns:
            dict mapping label -> predicted probability
        """
        # Noisy truthful agent predicts based on task prior
        # (Assumes others are also somewhat truthful)
        return {0: 1 - task.prior_prob, 1: task.prior_prob}
