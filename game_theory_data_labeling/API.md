# API Documentation

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Tasks](#tasks)
4. [Agents](#agents)
5. [Mechanisms](#mechanisms)
6. [Simulation](#simulation)
7. [Evaluation](#evaluation)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd game_theory_data_labeling

# Install dependencies (Python 3.10+)
pip install numpy scipy matplotlib seaborn pyyaml pytest

# Run tests
pytest tests/ -v

# Run experiments
python experiments/phase1_validation.py
python experiments/phase2_validation.py
python experiments/phase3_validation.py
```

### Minimal Example

```python
from src.simulation.game import Game, ExperimentConfig
import numpy as np

# Configure experiment
config = ExperimentConfig(
    name="my_experiment",
    num_tasks=20,
    num_agents=5,
    agent_mix={"truthful": 1.0},
    agent_params={"truthful": {"ability": 0.9}},
    mechanism_name="majority_voting",
    random_seed=42
)

# Run game
rng = np.random.default_rng(42)
game = Game(config, rng)
result = game.run()

# Evaluate
from src.evaluation.metrics import evaluate_game
metrics = evaluate_game(
    result.result.aggregated_labels,
    result.true_labels,
    result.result.payments
)

print(f"Accuracy: {metrics.accuracy:.3f}")
print(f"Total Payment: ${metrics.total_payment:.2f}")
```

---

## Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Experiment Config                      │
│  (num_tasks, num_agents, agent_mix, mechanism_name, etc.)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │  Game   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐      ┌────────┐     ┌──────────┐
    │ Tasks  │      │ Agents │     │Mechanism │
    └────────┘      └────────┘     └──────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ GameInstance    │
                │ (reports, etc.) │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Evaluation      │
                │ (metrics)       │
                └─────────────────┘
```

### Key Classes

- **Task**: Represents a labeling task with ground truth
- **Agent**: Makes observations and reports labels
- **Mechanism**: Aggregates reports and computes payments
- **Game**: Orchestrates single game instance
- **SimulationEngine**: Runs multiple games with statistics

---

## Tasks

### Task Class

```python
from src.tasks.base import Task

task = Task(
    task_id=0,
    true_label=1,        # Ground truth (0 or 1)
    difficulty=0.5,      # 0.0 (easy) to 1.0 (hard)
    prior_prob=0.5       # P(label=1)
)
```

### TaskGenerator

```python
from src.tasks.base import TaskGenerator

generator = TaskGenerator(
    num_tasks=50,
    label_prior=0.5,      # P(label=1)
    difficulty=0.5,       # Task difficulty
    random_seed=42
)

tasks = generator.generate()
```

**Fields**:
- `num_tasks`: Number of tasks to generate
- `label_prior`: Prior probability of positive class
- `difficulty`: Task difficulty (currently uniform, could be extended)
- `random_seed`: For reproducibility

---

## Agents

### Base Agent

All agents inherit from `Agent` base class:

```python
from src.agents.base import Agent, AgentParams
import numpy as np

params = AgentParams(
    agent_id=0,
    agent_type="truthful",
    ability=0.9,          # Observation accuracy
    effort_cost=0.0,      # Cost per unit effort
    risk_aversion=0.0     # For strategic agents
)

rng = np.random.default_rng(42)
```

### Agent Methods

**Required Methods**:
```python
def observe(task: Task) -> int:
    """Observe task and get (possibly noisy) signal."""
    pass

def report(task: Task, signal: int, mechanism: Mechanism,
           other_agents: list[Agent]) -> int:
    """Decide what to report."""
    pass

def predict_others(task: Task) -> dict[int, float]:
    """Predict distribution of others' reports (for RBTS)."""
    pass
```

### Agent Types

#### 1. Truthful Agent

```python
from src.agents.truthful import TruthfulAgent

agent = TruthfulAgent(params, rng)
```

**Behavior**:
- Observes with accuracy = `ability`
- Always reports observation truthfully

**Use Case**: Well-intentioned, skilled annotators

#### 2. Lazy Agent

```python
from src.agents.lazy import LazyAgent

agent = LazyAgent(params, rng, strategy="random")
# strategy: "random" or "prior"
```

**Behavior**:
- `random`: Reports uniformly at random
- `prior`: Reports based on task prior probability

**Use Case**: Low-effort workers, spammers

#### 3. Strategic Agent

```python
from src.agents.strategic import StrategicAgent

agent = StrategicAgent(params, rng)
```

**Behavior**:
- Observes with accuracy = `ability`
- Best-responds to mechanism incentives
- Maximizes expected payment

**Use Case**: Sophisticated, game-aware annotators

#### 4. Adversarial Agent

```python
from src.agents.adversarial import AdversarialAgent

agent = AdversarialAgent(params, rng, strategy="random")
# strategy: "always_wrong", "random", or "confuse"
```

**Behavior**:
- `always_wrong`: Reports opposite of observation
- `random`: Reports randomly
- `confuse`: Predicts majority and reports opposite

**Use Case**: Malicious actors, saboteurs

#### 5. Noisy Truthful Agent

```python
from src.agents.noisy_truthful import NoisyTruthfulAgent

agent = NoisyTruthfulAgent(params, rng)
```

**Behavior**:
- Observes with ability-based noise
- Always reports observation truthfully

**Use Case**: Well-intentioned but less skilled workers

---

## Mechanisms

### Base Mechanism

All mechanisms inherit from `Mechanism`:

```python
from src.mechanisms.base import Mechanism, Report, MechanismResult

class MyMechanism(Mechanism):
    def aggregate_and_pay(self, reports: list[Report],
                         tasks: list[Task]) -> MechanismResult:
        # Implement aggregation and payment logic
        pass

    def requires_predictions(self) -> bool:
        # True if mechanism needs agent predictions
        return False
```

### Report Structure

```python
from src.mechanisms.base import Report

report = Report(
    agent_id=0,
    task_id=0,
    report=1,                    # Agent's report (0 or 1)
    prediction={0: 0.3, 1: 0.7}  # Optional predictions (for RBTS)
)
```

### MechanismResult Structure

```python
from src.mechanisms.base import MechanismResult

result = MechanismResult(
    aggregated_labels={0: 1, 1: 0, ...},  # task_id -> predicted label
    payments={0: 10.5, 1: 12.3, ...},     # agent_id -> payment
    agent_qualities={0: 0.95, 1: 0.87}    # Optional: agent_id -> quality
)
```

### Mechanism Types

#### 1. Majority Voting

```python
from src.mechanisms.majority_voting import MajorityVoting

mechanism = MajorityVoting({
    "payment_per_task": 1.0
})
```

**Algorithm**:
- Aggregation: Simple majority vote
- Payment: Fixed per task

**Properties**:
- O(n) complexity
- No quality weighting
- Transparent, simple

#### 2. Dawid-Skene

```python
from src.mechanisms.dawid_skene import DawidSkene

mechanism = DawidSkene({
    "payment_per_task": 1.0,
    "quality_weight": 0.5,    # Weight for quality bonus (0-1)
    "max_iterations": 100,
    "tolerance": 1e-4
})
```

**Algorithm**:
- Aggregation: EM algorithm for joint inference
- Payment: Fixed + quality-weighted bonus
- Estimates: Agent confusion matrices

**Properties**:
- O(kTN) complexity where k = EM iterations
- Models agent-specific accuracy
- Robust to low-quality agents

#### 3. Output Agreement

```python
from src.mechanisms.output_agreement import OutputAgreement

mechanism = OutputAgreement({
    "agreement_payment": 1.0,
    "disagreement_payment": 0.0,
    "seed": 42
})
```

**Algorithm**:
- Aggregation: Majority vote
- Payment: Bonus for agreeing with random peer

**Properties**:
- O(n) complexity
- Incentivizes consensus
- Game-theoretic

#### 4. RBTS (Robust Bayesian Truth Serum)

```python
from src.mechanisms.rbts import RBTS

mechanism = RBTS({
    "base_payment": 1.0,
    "bonus_scale": 0.5
})
```

**Algorithm**:
- Aggregation: Majority vote
- Payment: Rewards "surprisingly common" answers
- Scoring: log(frequency / prediction)

**Properties**:
- O(n) complexity
- Requires agent predictions
- Theoretically incentive-compatible
- Detail-free (no common prior needed)

---

## Simulation

### ExperimentConfig

```python
from src.simulation.game import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    num_tasks=50,
    num_agents=10,
    agent_mix={
        "truthful": 0.6,
        "lazy": 0.2,
        "strategic": 0.2
    },
    agent_params={
        "truthful": {"ability": 0.9},
        "strategic": {"ability": 0.85}
    },
    mechanism_name="dawid_skene",
    mechanism_params={
        "payment_per_task": 1.0,
        "quality_weight": 0.5
    },
    task_difficulty=0.5,
    label_prior=0.5,
    num_runs=10,
    random_seed=42
)
```

**Fields**:
- `name`: Experiment identifier
- `num_tasks`: Number of tasks per game
- `num_agents`: Total number of agents
- `agent_mix`: Dict of agent_type -> proportion
- `agent_params`: Dict of agent_type -> {parameter: value}
- `mechanism_name`: "majority_voting", "dawid_skene", "output_agreement", or "rbts"
- `mechanism_params`: Mechanism-specific parameters
- `task_difficulty`: Task difficulty level
- `label_prior`: P(label=1)
- `num_runs`: Number of simulation runs
- `random_seed`: Master random seed

### Game Class

```python
from src.simulation.game import Game
import numpy as np

rng = np.random.default_rng(42)
game = Game(config, rng)
result = game.run()

# Access results
print(result.result.aggregated_labels)  # Predicted labels
print(result.result.payments)           # Agent payments
print(result.true_labels)               # Ground truth
```

### SimulationEngine

```python
from src.simulation.engine import SimulationEngine

engine = SimulationEngine(config)
results = engine.run_experiment()

# Access aggregated statistics
print(f"Mean accuracy: {results.mean_metrics['accuracy']:.3f}")
print(f"Std accuracy:  {results.std_metrics['accuracy']:.3f}")
print(f"95% CI:        {results.confidence_intervals['accuracy']}")
```

**Results Fields**:
- `mean_metrics`: Mean of each metric across runs
- `std_metrics`: Standard deviation
- `confidence_intervals`: 95% confidence intervals
- `metrics`: List of individual run metrics

---

## Evaluation

### Metrics

```python
from src.evaluation.metrics import evaluate_game, MetricResult

metrics = evaluate_game(
    predicted_labels={0: 1, 1: 0, ...},
    true_labels={0: 1, 1: 1, ...},
    payments={0: 10.5, 1: 12.3, ...}
)

print(f"Accuracy:         {metrics.accuracy:.3f}")
print(f"F1 Score:         {metrics.f1_score:.3f}")
print(f"Precision:        {metrics.precision:.3f}")
print(f"Recall:           {metrics.recall:.3f}")
print(f"Total Payment:    ${metrics.total_payment:.2f}")
print(f"Average Payment:  ${metrics.average_payment:.2f}")
print(f"Payment Std:      ${metrics.payment_std:.2f}")
print(f"Quality/Dollar:   {metrics.quality_per_dollar:.4f}")
```

### Helper Functions

```python
from src.evaluation.metrics import (
    compute_accuracy,
    compute_precision_recall_f1
)

accuracy = compute_accuracy(predicted, true_labels)
precision, recall, f1 = compute_precision_recall_f1(predicted, true_labels)
```

---

## Configuration

### Agent Mix Examples

```python
# All truthful
agent_mix = {"truthful": 1.0}

# Mixed quality
agent_mix = {"truthful": 0.7, "lazy": 0.3}

# Realistic population
agent_mix = {
    "truthful": 0.5,
    "lazy": 0.3,
    "strategic": 0.1,
    "noisy_truthful": 0.1
}

# Adversarial scenario
agent_mix = {"truthful": 0.6, "adversarial": 0.4}
```

### Agent Parameters Examples

```python
# High-ability truthful agents
agent_params = {"truthful": {"ability": 0.95}}

# Mixed abilities
agent_params = {
    "truthful": {"ability": 0.9},
    "strategic": {"ability": 0.85},
    "noisy_truthful": {"ability": 0.7}
}
```

---

## Examples

### Example 1: Compare Mechanisms

```python
from src.simulation.engine import SimulationEngine
from src.simulation.game import ExperimentConfig

mechanisms = ["majority_voting", "dawid_skene", "output_agreement", "rbts"]

for mech in mechanisms:
    config = ExperimentConfig(
        name=f"compare_{mech}",
        num_tasks=50,
        num_agents=10,
        agent_mix={"truthful": 0.7, "lazy": 0.3},
        mechanism_name=mech,
        num_runs=20,
        random_seed=42
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    print(f"{mech}: {results.mean_metrics['accuracy']:.3f}")
```

### Example 2: Test Adversarial Robustness

```python
adversarial_percentages = [0.0, 0.2, 0.4, 0.5]

for adv_pct in adversarial_percentages:
    config = ExperimentConfig(
        name=f"adversarial_{int(adv_pct*100)}",
        num_tasks=50,
        num_agents=10,
        agent_mix={
            "truthful": 1.0 - adv_pct,
            "adversarial": adv_pct
        },
        mechanism_name="dawid_skene",
        num_runs=15,
        random_seed=42
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    print(f"{int(adv_pct*100)}% adversarial: {results.mean_metrics['accuracy']:.3f}")
```

### Example 3: Custom Mechanism

```python
from src.mechanisms.base import Mechanism, Report, MechanismResult
from src.tasks.base import Task

class WeightedVoting(Mechanism):
    """Weighted voting based on historical accuracy."""

    def __init__(self, config: dict | None = None):
        super().__init__(config or {})
        self.weights = self.config.get("weights", {})

    def requires_predictions(self) -> bool:
        return False

    def aggregate_and_pay(
        self, reports: list[Report], tasks: list[Task]
    ) -> MechanismResult:
        reports_by_task = self._group_by_task(reports)
        aggregated_labels = {}

        for task_id, task_reports in reports_by_task.items():
            weighted_sum = sum(
                r.report * self.weights.get(r.agent_id, 1.0)
                for r in task_reports
            )
            total_weight = sum(
                self.weights.get(r.agent_id, 1.0)
                for r in task_reports
            )
            aggregated_labels[task_id] = 1 if weighted_sum > total_weight / 2 else 0

        payments = {r.agent_id: 1.0 for r in reports}

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=None
        )

# Use custom mechanism
mechanism = WeightedVoting({"weights": {0: 2.0, 1: 1.5, 2: 1.0}})
```

---

## Advanced Usage

### Seed Management

```python
# Master seed for reproducibility
master_seed = 42

# Game-specific seed
game_seed = master_seed + run_id

# Agent-specific seed
agent_seed = game_seed + 1000 * agent_id
```

### Logging Results

```python
import json

# Save configuration
with open("config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

# Save results
results_dict = {
    "mean_metrics": results.mean_metrics,
    "std_metrics": results.std_metrics,
    "confidence_intervals": results.confidence_intervals
}
with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mechanisms.py -v

# Run specific test
pytest tests/test_mechanisms.py::test_dawid_skene_convergence -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Extending the Framework

### Add New Agent Type

1. Create file in `src/agents/`
2. Inherit from `Agent`
3. Implement `observe()`, `report()`, and `predict_others()`
4. Add to `src/agents/__init__.py`
5. Update `Game._create_agent()` in `src/simulation/game.py`
6. Write tests in `tests/`

### Add New Mechanism

1. Create file in `src/mechanisms/`
2. Inherit from `Mechanism`
3. Implement `aggregate_and_pay()` and `requires_predictions()`
4. Add to `src/mechanisms/__init__.py`
5. Update `Game._create_mechanism()` in `src/simulation/game.py`
6. Write tests in `tests/`

---

## Troubleshooting

### Common Issues

**Issue**: EM algorithm not converging
- **Solution**: Increase `max_iterations` or adjust `tolerance`

**Issue**: RBTS raises "missing predictions" error
- **Solution**: Ensure all agents implement `predict_others()`

**Issue**: Inconsistent results across runs
- **Solution**: Check random seed settings

**Issue**: Low accuracy with high-ability agents
- **Solution**: Verify agent parameters are correctly specified

---

## Performance Tips

1. **Batch processing**: Run multiple configurations in parallel
2. **Reduced runs**: Start with fewer runs (5-10) for quick testing
3. **Smaller experiments**: Test with fewer tasks/agents initially
4. **Profiling**: Use Python profiler for bottlenecks

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{game_theory_data_labeling,
  title = {Game-Theoretic Mechanisms for Crowdsourced Data Labeling},
  author = {[Your Name]},
  year = {2025},
  url = {[repository-url]}
}
```

---

*For more examples, see `experiments/` directory*
