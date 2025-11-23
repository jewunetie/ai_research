"""Base agent classes."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.tasks.base import Task
    from src.mechanisms.base import Mechanism


@dataclass
class AgentParams:
    """Parameters defining an agent's characteristics."""

    agent_id: int
    agent_type: Literal["truthful", "lazy", "strategic", "adversarial", "noisy_truthful"]
    ability: float  # 0.5 (random) to 1.0 (perfect) - accuracy when trying
    effort_cost: float = 0.0  # Cost per unit of effort
    risk_aversion: float = 0.0  # For strategic agents


class Agent(ABC):
    """Base class for all agent types."""

    def __init__(self, params: AgentParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    @abstractmethod
    def observe(self, task: "Task") -> int:
        """Observe the task and get a (possibly noisy) signal.

        Args:
            task: The task to observe

        Returns:
            signal: 0 or 1 (agent's observation)
        """
        pass

    @abstractmethod
    def report(
        self,
        task: "Task",
        signal: int,
        mechanism: "Mechanism",
        other_agents: list["Agent"],
    ) -> int:
        """Decide what to report given observation and mechanism.

        Args:
            task: The task to label
            signal: Agent's observation (from self.observe)
            mechanism: The mechanism being used
            other_agents: Other agents in the game (for strategic reasoning)

        Returns:
            report: 0 or 1 (what agent reports)
        """
        pass

    def predict_others(self, task: "Task") -> dict[int, float]:
        """Predict distribution of others' reports (for BTS/RBTS).

        Args:
            task: The task

        Returns:
            dict mapping label -> predicted probability
        """
        # Default: uniform prediction
        return {0: 0.5, 1: 0.5}
