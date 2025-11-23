"""Game simulation classes."""

from dataclasses import dataclass
from typing import Literal
import numpy as np

from src.tasks.base import Task, TaskGenerator
from src.agents.base import Agent, AgentParams
from src.agents.truthful import TruthfulAgent
from src.agents.lazy import LazyAgent
from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.mechanisms.majority_voting import MajorityVoting
from src.mechanisms.dawid_skene import DawidSkene


@dataclass
class GameInstance:
    """A single instance of the labeling game."""

    game_id: int
    tasks: list[Task]
    agents: list[Agent]
    mechanism: Mechanism
    reports: list[Report]
    result: MechanismResult
    true_labels: dict[int, int]  # task_id -> true label


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    num_tasks: int
    num_agents: int
    agent_mix: dict[str, float]  # type -> proportion
    agent_params: dict[str, dict] | None = None  # type -> {ability, cost, etc.}
    mechanism_name: str = "majority_voting"
    mechanism_params: dict | None = None
    task_difficulty: float = 0.5
    label_prior: float = 0.5
    num_runs: int = 1
    random_seed: int = 42


class Game:
    """A single instance of the labeling game."""

    def __init__(self, config: ExperimentConfig, rng: np.random.Generator):
        """
        Args:
            config: Experiment configuration
            rng: Random number generator for this game
        """
        self.config = config
        self.rng = rng

        # Generate tasks
        self.tasks = self._generate_tasks()

        # Generate agents
        self.agents = self._generate_agents()

        # Initialize mechanism
        self.mechanism = self._create_mechanism()

    def _generate_tasks(self) -> list[Task]:
        """Generate tasks according to config."""
        generator = TaskGenerator(
            num_tasks=self.config.num_tasks,
            label_prior=self.config.label_prior,
            difficulty=self.config.task_difficulty,
            random_seed=int(self.rng.integers(0, 2**31)),
        )
        return generator.generate()

    def _generate_agents(self) -> list[Agent]:
        """Generate agents according to config."""
        agents = []
        agent_id = 0

        for agent_type, proportion in self.config.agent_mix.items():
            num_agents = int(self.config.num_agents * proportion)

            for _ in range(num_agents):
                # Get agent parameters from config or use defaults
                if self.config.agent_params and agent_type in self.config.agent_params:
                    type_params = self.config.agent_params[agent_type]
                else:
                    type_params = {}

                # Create agent params with defaults
                params = AgentParams(
                    agent_id=agent_id,
                    agent_type=agent_type,  # type: ignore
                    ability=type_params.get("ability", 0.8),
                    effort_cost=type_params.get("effort_cost", 0.0),
                    risk_aversion=type_params.get("risk_aversion", 0.0),
                )

                # Create agent RNG
                agent_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))

                # Create agent instance
                agent = self._create_agent(agent_type, params, agent_rng)
                agents.append(agent)
                agent_id += 1

        return agents

    def _create_agent(
        self, agent_type: str, params: AgentParams, rng: np.random.Generator
    ) -> Agent:
        """Create an agent of the specified type."""
        if agent_type == "truthful":
            return TruthfulAgent(params, rng)
        elif agent_type == "lazy":
            return LazyAgent(params, rng, strategy="random")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _create_mechanism(self) -> Mechanism:
        """Create mechanism from config."""
        config = self.config.mechanism_params or {}

        if self.config.mechanism_name == "majority_voting":
            return MajorityVoting(config)
        elif self.config.mechanism_name == "dawid_skene":
            return DawidSkene(config)
        else:
            raise ValueError(f"Unknown mechanism: {self.config.mechanism_name}")

    def run(self) -> GameInstance:
        """Run the game: agents observe, report, mechanism aggregates.

        IMPORTANT: Reporting is SIMULTANEOUS, not sequential.
        - All agents observe and decide reports without seeing others' actual reports
        - The loop is for implementation convenience only
        - Passing self.agents allows strategic reasoning about agent types/distribution
        - Strategic agents can reason about what others WILL report, but don't see actual reports
        - Payments are computed after all reports are collected
        """
        reports = []

        # For each task, get reports from agents
        # NOTE: This loop processes agents sequentially, but conceptually they report simultaneously
        for task in self.tasks:
            for agent in self.agents:
                # Agent observes their private signal
                signal = agent.observe(task)

                # Agent decides what to report (simultaneously, without seeing others' reports)
                # Passing self.agents allows reasoning about agent types, NOT seeing their reports
                report_value = agent.report(task, signal, self.mechanism, self.agents)

                # If mechanism requires predictions, get prediction
                prediction = None
                if self.mechanism.requires_predictions():
                    prediction = agent.predict_others(task)

                # Record report
                reports.append(
                    Report(
                        agent_id=agent.params.agent_id,
                        task_id=task.task_id,
                        report=report_value,
                        prediction=prediction,
                    )
                )

        # Mechanism aggregates and computes payments
        result = self.mechanism.aggregate_and_pay(reports, self.tasks)

        # Create game instance
        true_labels = {t.task_id: t.true_label for t in self.tasks}

        return GameInstance(
            game_id=self.config.random_seed,
            tasks=self.tasks,
            agents=self.agents,
            mechanism=self.mechanism,
            reports=reports,
            result=result,
            true_labels=true_labels,
        )
