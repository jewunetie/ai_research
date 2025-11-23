"""Adversarial agent implementation."""

from typing import Literal, TYPE_CHECKING
import numpy as np
from src.agents.base import Agent, AgentParams
from src.tasks.base import Task

if TYPE_CHECKING:
    from src.mechanisms.base import Mechanism


class AdversarialAgent(Agent):
    """Agent that tries to game the system.

    This agent:
    - May observe correctly (depending on strategy)
    - Reports strategically to maximize payment while hurting quality
    - Represents malicious annotators, spammers, saboteurs
    """

    def __init__(
        self,
        params: AgentParams,
        rng: np.random.Generator,
        strategy: Literal["always_wrong", "random", "confuse"] = "random",
    ):
        """
        Args:
            params: Agent parameters
            rng: Random number generator
            strategy: Adversarial strategy
                - 'always_wrong': Report opposite of signal
                - 'random': Report randomly (unpredictable)
                - 'confuse': Try to disagree with majority
        """
        super().__init__(params, rng)
        self.strategy = strategy

    def observe(self, task: Task) -> int:
        """Adversarial agent may observe correctly to game mechanism.

        Args:
            task: The task to observe

        Returns:
            signal: Observed label (0 or 1)
        """
        # Some adversarial strategies need true signal
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
        """Report to maximize payment while hurting quality.

        Args:
            task: The task to label
            signal: Agent's observation
            mechanism: The mechanism (ignored by most strategies)
            other_agents: Other agents (for 'confuse' strategy)

        Returns:
            report: Adversarial report (0 or 1)
        """
        if self.strategy == "always_wrong":
            # Always report opposite of signal
            return 1 - signal

        elif self.strategy == "random":
            # Report randomly to be unpredictable
            return self.rng.integers(0, 2)

        elif self.strategy == "confuse":
            # Try to predict majority and report opposite
            # This hurts consensus-based mechanisms
            predicted_majority = self._predict_majority(task, other_agents)
            return 1 - predicted_majority

        else:
            raise ValueError(f"Unknown adversarial strategy: {self.strategy}")

    def _predict_majority(self, task: Task, others: list[Agent]) -> int:
        """Predict what majority will report.

        Args:
            task: The task
            others: Other agents

        Returns:
            predicted_majority: 0 or 1
        """
        # Simple heuristic: assume majority follows prior
        return 1 if task.prior_prob > 0.5 else 0

    def predict_others(self, task: Task) -> dict[int, float]:
        """Predict distribution of others' reports (for RBTS).

        Args:
            task: The task

        Returns:
            dict mapping label -> predicted probability
        """
        # Adversarial agent may lie about predictions
        # For now: uniform (doesn't help or hurt)
        return {0: 0.5, 1: 0.5}
