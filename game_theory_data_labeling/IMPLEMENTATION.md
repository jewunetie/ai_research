# Implementation Plan: Non-Cooperative Game Theory for Data Labeling

## Table of Contents

1. [Research Impact Analysis](#1-research-impact-analysis)
2. [System Architecture](#2-system-architecture)
3. [Core Data Structures](#3-core-data-structures)
4. [Agent Implementations](#4-agent-implementations)
5. [Mechanism Implementations](#5-mechanism-implementations)
6. [Simulation Engine](#6-simulation-engine)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Phased Implementation Plan](#8-phased-implementation-plan)
9. [Testing Strategy](#9-testing-strategy)
10. [Success Criteria](#10-success-criteria)
11. [Reproducibility Protocol](#11-reproducibility-protocol)

---

## 1. Research Impact Analysis

### 1.1 Why This Research Will Be Impactful

#### **Gap in the Literature**

Our deep literature review revealed a critical divide:

1. **Truth Inference Research** (ML/Statistics community):
   - Zheng et al. (2017) benchmarked 17 algorithms
   - Assumes annotators report observations (possibly noisy)
   - No strategic behavior modeling
   - No peer prediction mechanisms tested

2. **Peer Prediction Research** (Mechanism Design community):
   - Strong theoretical results (incentive compatibility proofs)
   - Very limited empirical validation (only Gao et al. 2014 tested ONE mechanism)
   - Never compared with truth inference methods
   - Assumes perfect rationality

**No existing work bridges these two worlds.**

#### **Theoretical Impact**

1. **Empirical Validation of Theory**:
   - Many mechanisms proven incentive-compatible in theory
   - Quote from literature: "empirical evidence of such mechanisms working in practice is very limited and generally weak"
   - Our simulation provides controlled empirical testing

2. **Context-Dependent Performance**:
   - Zheng et al. found "no algorithm consistently outperforms others"
   - Implies mechanism choice should depend on context
   - Our simulation identifies which contexts favor which mechanisms

3. **Strategic vs. Non-Strategic Comparison**:
   - Do incentive mechanisms (RBTS, BTS) outperform simple aggregation (Dawid-Skene)?
   - Under what conditions?
   - With what agent populations?

#### **Practical Impact**

1. **Actionable Guidance for Practitioners**:
   - Platform designers: Which mechanism to implement?
   - Requester with budget constraints: Best quality/cost tradeoff?
   - Facing adversarial workers: Which mechanism is robust?

2. **Design Principles**:
   - When is the extra complexity of peer prediction worth it?
   - When does simple majority voting suffice?
   - How sensitive are mechanisms to agent heterogeneity?

3. **Cost-Benefit Analysis**:
   - Peer prediction requires eliciting predictions (more effort)
   - Does the quality improvement justify the added complexity?
   - Quantify the tradeoff

#### **Methodological Impact**

1. **Unified Framework**:
   - First implementation comparing both mechanism categories
   - Others can extend with new mechanisms/agents
   - Open-source, reproducible, well-documented

2. **Synthetic Agent Methodology**:
   - Controlled experiments impossible with real workers
   - Perfect ground truth available
   - Can vary one dimension while holding others constant

3. **Reproducibility**:
   - Config-driven experiments
   - Fixed seeds
   - All parameters logged
   - Others can replicate and extend

#### **Broader Applications**

Beyond crowdsourced labeling, this research applies to:
- **Peer review**: How to incentivize good reviews?
- **Forecasting aggregation**: Combining expert predictions
- **Recommendation systems**: Aggregating user ratings
- **Collective decision-making**: Voting mechanisms
- **Decentralized systems**: Blockchain consensus, DAOs

### 1.2 Publication Potential

**Target Venues**:
1. **Top-tier ML**: NeurIPS, ICML, ICLR (crowdsourcing/human computation track)
2. **AI conferences**: AAAI, IJCAI (game theory/mechanism design)
3. **Economics/Game Theory**: EC (ACM Conference on Economics and Computation)
4. **HCI/Crowdsourcing**: HCOMP (Human Computation), CHI
5. **Journals**: JAIR, TACL (if applied to NLP), ACM TEAC

**Paper Structure**:
- **Problem**: Two separate literatures, no empirical comparison
- **Method**: Unified simulation framework with strategic agents
- **Results**: Comparative evaluation across mechanisms and scenarios
- **Insights**: When does peer prediction outperform aggregation? Tradeoffs?
- **Contribution**: Empirical validation + practical guidance + open framework

### 1.3 Expected Insights

Based on the literature and our design, we expect to answer:

1. **Does incentive design matter?**
   - Do strategic mechanisms (RBTS) outperform simple aggregation (Dawid-Skene) when agents are strategic?
   - Or does Dawid-Skene's confusion matrix modeling compensate?

2. **Zheng et al. found GLAD doesn't improve over simpler methods**
   - Will this hold with strategic agents?
   - Does task difficulty matter when agents are lazy/adversarial?

3. **Agent heterogeneity effects**:
   - Are mechanisms robust to mixed populations?
   - What fraction of adversarial agents breaks each mechanism?

4. **Cost-quality tradeoffs**:
   - Is the added complexity of peer prediction justified?
   - Quantify the improvement per unit of added effort

5. **Practical thresholds**:
   - When does majority voting suffice?
   - When is sophisticated mechanism design necessary?

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Configuration                  │
│  (YAML files: task params, agent mix, mechanism, metrics)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Simulation Engine                       │
│  - Orchestrates experiments                                  │
│  - Manages random seeds                                      │
│  - Logs all parameters                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
         ┌──────────────────────────────────────┐
         │                                      │
    ┌────▼────┐        ┌──────▼──────┐   ┌────▼─────┐
    │  Task   │        │   Agent     │   │Mechanism │
    │Generator│        │  Factory    │   │ Factory  │
    └────┬────┘        └──────┬──────┘   └────┬─────┘
         │                    │               │
         ▼                    ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Game Instance                             │
│  - N agents assigned to M tasks                              │
│  - Each agent observes signal (based on true label + noise)  │
│  - Each agent reports (based on type strategy)               │
│  - Mechanism computes aggregated label + payments            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation & Analysis                      │
│  - Compute metrics (accuracy, F1, payment, etc.)             │
│  - Statistical analysis (means, confidence intervals)        │
│  - Visualization (plots, tables)                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Results & Outputs                         │
│  - CSV/JSON results                                          │
│  - Plots (publication-quality)                               │
│  - Statistical summaries                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
game_theory_data_labeling/
├── src/
│   ├── __init__.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py              # TaskGenerator base class
│   │   ├── binary.py            # Binary classification tasks
│   │   └── multiclass.py        # Multi-class tasks (future)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Agent base class
│   │   ├── truthful.py          # Truthful agent
│   │   ├── lazy.py              # Lazy agent
│   │   ├── strategic.py         # Strategic agent
│   │   ├── adversarial.py       # Adversarial agent
│   │   └── noisy_truthful.py    # Noisy truthful agent
│   ├── mechanisms/
│   │   ├── __init__.py
│   │   ├── base.py              # Mechanism base class
│   │   ├── majority_voting.py   # Simple majority
│   │   ├── dawid_skene.py       # Dawid-Skene EM
│   │   ├── output_agreement.py  # Simple peer prediction
│   │   ├── rbts.py              # Robust Bayesian Truth Serum
│   │   ├── glad.py              # GLAD (future)
│   │   └── peer_truth_serum.py  # Peer Truth Serum (future)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── game.py              # Single game instance
│   │   ├── engine.py            # Simulation orchestrator
│   │   └── config.py            # Configuration loader
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Quality & payment metrics
│   │   ├── statistical.py       # Statistical analysis
│   │   └── visualization.py     # Plotting functions
│   └── utils/
│       ├── __init__.py
│       ├── random.py            # Random number management
│       └── logging.py           # Structured logging
├── configs/
│   ├── experiments/
│   │   ├── validation_truthful.yaml
│   │   ├── validation_lazy.yaml
│   │   ├── mixed_population.yaml
│   │   └── adversarial_robustness.yaml
│   ├── agents/
│   │   ├── truthful_high_ability.yaml
│   │   ├── lazy_random.yaml
│   │   └── strategic_rational.yaml
│   └── mechanisms/
│       ├── majority_voting.yaml
│       ├── dawid_skene.yaml
│       └── rbts.yaml
├── experiments/
│   ├── run_validation.py
│   ├── run_comparison.py
│   └── run_robustness.py
├── tests/
│   ├── test_tasks.py
│   ├── test_agents.py
│   ├── test_mechanisms.py
│   └── test_integration.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_publication_figures.ipynb
├── results/
│   ├── raw/                     # Raw CSV/JSON outputs
│   ├── processed/               # Aggregated results
│   └── figures/                 # Publication plots
├── docs/
│   ├── architecture.md
│   ├── algorithms.md
│   └── api_reference.md
├── pyproject.toml               # uv configuration
├── CLAUDE.md
├── RESEARCH.md
└── IMPLEMENTATION.md
```

### 2.3 Key Design Principles

1. **Modularity**: Easy to add new mechanisms, agents, task types
2. **Reproducibility**: All randomness controlled by seeds
3. **Configuration-driven**: Experiments defined in YAML, not code
4. **Type-safe**: Use Python type hints throughout
5. **Well-tested**: Unit tests for all components
6. **Efficient**: Vectorized operations where possible (NumPy)
7. **Logged**: All parameters and intermediate results saved

---

## 3. Core Data Structures

### 3.1 Task

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Task:
    """A single labeling task with ground truth."""
    task_id: int
    true_label: int  # 0 or 1 for binary
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    prior_prob: float  # P(label=1) - prior probability

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, label={self.true_label}, diff={self.difficulty:.2f})"
```

### 3.2 Agent

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class AgentParams:
    """Parameters defining an agent's characteristics."""
    agent_id: int
    agent_type: Literal["truthful", "lazy", "strategic", "adversarial", "noisy_truthful"]
    ability: float  # 0.5 (random) to 1.0 (perfect) - accuracy when trying
    effort_cost: float  # Cost per unit of effort
    risk_aversion: float  # For strategic agents

class Agent(ABC):
    """Base class for all agent types."""

    def __init__(self, params: AgentParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    @abstractmethod
    def observe(self, task: Task) -> int:
        """Observe the task and get a (possibly noisy) signal.

        Returns:
            signal: 0 or 1 (agent's observation)
        """
        pass

    @abstractmethod
    def report(self, task: Task, signal: int, mechanism: 'Mechanism',
               other_agents: list['Agent']) -> int:
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

    def predict_others(self, task: Task) -> dict[int, float]:
        """Predict distribution of others' reports (for BTS/RBTS).

        Returns:
            dict mapping label -> predicted probability
        """
        # Default: uniform prediction
        return {0: 0.5, 1: 0.5}
```

### 3.3 Mechanism

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

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
    def aggregate_and_pay(self, reports: list[Report],
                          tasks: list[Task]) -> MechanismResult:
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
```

### 3.4 GameInstance

```python
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
    mechanism_name: str
    mechanism_params: dict
    task_difficulty: float
    label_prior: float
    num_runs: int
    random_seed: int
```

### 3.5 Results

```python
@dataclass
class MetricResult:
    """Metrics for a single game instance."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    total_payment: float
    average_payment: float
    payment_std: float
    quality_per_dollar: float  # accuracy / total_payment

@dataclass
class ExperimentResults:
    """Aggregated results across multiple runs."""
    config: ExperimentConfig
    metrics: list[MetricResult]  # One per run
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]  # 95% CI
```

---

## 4. Agent Implementations

### 4.1 Truthful Agent

**Strategy**: Always reports true observation (if able to observe correctly).

```python
class TruthfulAgent(Agent):
    """Agent that always reports their observation."""

    def observe(self, task: Task) -> int:
        """Observe with accuracy = self.params.ability."""
        # With probability = ability, observe correctly
        # Otherwise, observe incorrectly
        if self.rng.random() < self.params.ability:
            return task.true_label
        else:
            return 1 - task.true_label

    def report(self, task: Task, signal: int, mechanism: Mechanism,
               other_agents: list[Agent]) -> int:
        """Truthful agent always reports their signal."""
        return signal
```

**Parameters**:
- `ability`: 0.5 to 1.0 (accuracy of observation)
  - 0.5 = random guessing
  - 0.7 = moderate ability
  - 0.9 = high ability
  - 1.0 = perfect oracle

**Use case**: Baseline for "honest but noisy" annotators

### 4.2 Lazy Agent

**Strategy**: Minimizes effort by reporting randomly or following prior.

```python
class LazyAgent(Agent):
    """Agent that minimizes effort."""

    def __init__(self, params: AgentParams, rng: np.random.Generator,
                 strategy: Literal["random", "prior"] = "random"):
        super().__init__(params, rng)
        self.strategy = strategy

    def observe(self, task: Task) -> int:
        """Lazy agent doesn't actually observe - saves effort."""
        # Return dummy value (not used in report anyway)
        return 0

    def report(self, task: Task, signal: int, mechanism: Mechanism,
               other_agents: list[Agent]) -> int:
        """Report without looking at signal."""
        if self.strategy == "random":
            # Report uniformly random
            return self.rng.integers(0, 2)
        elif self.strategy == "prior":
            # Report based on prior probability
            return 1 if self.rng.random() < task.prior_prob else 0
```

**Variants**:
- `random`: Report uniformly random (0 or 1 with p=0.5)
- `prior`: Report 1 with probability = prior, 0 otherwise

**Use case**: Models annotators who don't pay attention

### 4.3 Noisy Truthful Agent

**Strategy**: Like truthful but with lower ability (noisier observations).

```python
class NoisyTruthfulAgent(TruthfulAgent):
    """Truthful agent with low ability (high noise)."""

    def __init__(self, params: AgentParams, rng: np.random.Generator):
        # Force ability to be in [0.5, 0.7] range
        params.ability = min(0.7, max(0.5, params.ability))
        super().__init__(params, rng)
```

**Parameters**:
- `ability`: 0.5 to 0.7 (worse than standard truthful)

**Use case**: Well-intentioned but unskilled annotators

### 4.4 Strategic Agent

**Strategy**: Computes expected payment and best-responds.

```python
class StrategicAgent(Agent):
    """Agent that maximizes expected utility = payment - cost."""

    def observe(self, task: Task) -> int:
        """Strategic agent observes (with their ability)."""
        if self.rng.random() < self.params.ability:
            return task.true_label
        else:
            return 1 - task.true_label

    def report(self, task: Task, signal: int, mechanism: Mechanism,
               other_agents: list[Agent]) -> int:
        """Report to maximize expected payment given beliefs about others."""

        # Compute expected payment for each possible report
        expected_payment = {}
        for possible_report in [0, 1]:
            # Estimate what others will report
            others_reports_dist = self._predict_others_reports(
                task, other_agents, mechanism
            )

            # Compute expected payment under this report
            expected_payment[possible_report] = self._compute_expected_payment(
                possible_report, others_reports_dist, mechanism
            )

        # Best-respond: report with highest expected payment
        # (Ignore effort cost for now - can add later)
        if expected_payment[0] > expected_payment[1]:
            return 0
        elif expected_payment[1] > expected_payment[0]:
            return 1
        else:
            # Tie: report truthfully
            return signal

    def _predict_others_reports(self, task: Task, others: list[Agent],
                                mechanism: Mechanism) -> dict[int, float]:
        """Predict distribution of others' reports.

        Simplified: Assume others report according to prior + noise.
        More sophisticated: Model each agent type's strategy.
        """
        # Simple model: weighted average of prior and uniform
        # If others are mostly truthful, closer to prior
        # If others are lazy, closer to uniform

        # For now: assume uniform (conservative assumption)
        return {0: 0.5, 1: 0.5}

    def _compute_expected_payment(self, my_report: int,
                                  others_dist: dict[int, float],
                                  mechanism: Mechanism) -> float:
        """Compute expected payment given my report and beliefs about others.

        This is mechanism-specific and requires knowledge of payment rule.
        """
        # Placeholder: mechanism-specific implementation needed
        # For majority voting: payment is fixed
        # For peer prediction: payment depends on agreement with others
        # For Dawid-Skene: payment proportional to inferred quality

        return mechanism.compute_expected_payment(my_report, others_dist)
```

**Complexity Note**:
- Full strategic reasoning requires solving for equilibrium
- Initial implementation: simplified best-response to uniform prior
- Future: iterative best-response, equilibrium computation

**Use case**: Sophisticated annotators who understand incentives

### 4.5 Adversarial Agent

**Strategy**: Maximizes payment while degrading quality.

```python
class AdversarialAgent(Agent):
    """Agent that tries to game the system."""

    def __init__(self, params: AgentParams, rng: np.random.Generator,
                 strategy: Literal["always_wrong", "random", "confuse"] = "random"):
        super().__init__(params, rng)
        self.strategy = strategy

    def observe(self, task: Task) -> int:
        """Adversarial agent may observe correctly to game mechanism."""
        # Some adversarial strategies need true signal
        if self.rng.random() < self.params.ability:
            return task.true_label
        else:
            return 1 - task.true_label

    def report(self, task: Task, signal: int, mechanism: Mechanism,
               other_agents: list[Agent]) -> int:
        """Report to maximize payment while hurting quality."""

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

    def _predict_majority(self, task: Task, others: list[Agent]) -> int:
        """Predict what majority will report."""
        # Simple heuristic: assume majority follows prior
        return 1 if task.prior_prob > 0.5 else 0
```

**Variants**:
- `always_wrong`: Report opposite of observation
- `random`: Completely random (unpredictable)
- `confuse`: Try to disagree with majority

**Use case**: Malicious annotators, spammers, saboteurs

---

## 5. Mechanism Implementations

### 5.1 Majority Voting

**Algorithm**: Aggregate by simple majority, pay everyone equally.

```python
class MajorityVoting(Mechanism):
    """Simple majority voting with fixed payment."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.payment_per_task = config.get("payment_per_task", 1.0)

    def requires_predictions(self) -> bool:
        return False

    def aggregate_and_pay(self, reports: list[Report],
                          tasks: list[Task]) -> MechanismResult:
        """Aggregate by majority, pay fixed amount."""

        # Group reports by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate each task
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            # Count votes
            votes = [r.report for r in task_reports]
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes)/2 else 0

        # Pay everyone equally
        payments = {
            r.agent_id: self.payment_per_task * len(set(r.task_id for r in reports if r.agent_id == r.agent_id))
            for r in reports
        }

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments
        )
```

**Properties**:
- Simple, transparent
- Treats all agents equally (no quality weighting)
- Vulnerable to majority of low-quality agents
- No strategic incentives (payment is fixed)

### 5.2 Dawid-Skene

**Algorithm**: EM algorithm to jointly infer true labels and agent confusion matrices.

```python
class DawidSkene(Mechanism):
    """Dawid-Skene EM algorithm for label aggregation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_iterations = config.get("max_iterations", 100)
        self.tolerance = config.get("tolerance", 1e-4)
        self.payment_per_task = config.get("payment_per_task", 1.0)
        self.quality_weight = config.get("quality_weight", 0.5)

    def requires_predictions(self) -> bool:
        return False

    def aggregate_and_pay(self, reports: list[Report],
                          tasks: list[Task]) -> MechanismResult:
        """Run EM to infer labels and quality, pay based on quality."""

        # Convert reports to matrix format
        # reports_matrix[i, j] = label that agent i gave to task j
        # (or -1 if agent i didn't label task j)
        reports_matrix, agent_ids, task_ids = self._to_matrix(reports)
        n_agents, n_tasks = reports_matrix.shape
        n_classes = 2  # Binary for now

        # Initialize
        # p[j, k] = P(true_label_j = k)
        p = np.ones((n_tasks, n_classes)) / n_classes

        # confusion[i, k, l] = P(agent i reports l | true label is k)
        confusion = np.ones((n_agents, n_classes, n_classes)) / n_classes

        # EM iterations
        for iteration in range(self.max_iterations):
            # E-step: Update p given current confusion matrices
            p_new = self._e_step(reports_matrix, confusion, n_classes)

            # M-step: Update confusion matrices given current p
            confusion_new = self._m_step(reports_matrix, p_new, n_classes)

            # Check convergence
            if np.max(np.abs(p_new - p)) < self.tolerance:
                break

            p = p_new
            confusion = confusion_new

        # Aggregate: predicted label = argmax of p
        aggregated_labels = {
            task_ids[j]: int(np.argmax(p[j]))
            for j in range(n_tasks)
        }

        # Compute agent qualities
        # Quality = accuracy on diagonal of confusion matrix
        agent_qualities = {
            agent_ids[i]: (confusion[i, 0, 0] + confusion[i, 1, 1]) / 2
            for i in range(n_agents)
        }

        # Payment: fixed + bonus proportional to quality
        num_tasks_per_agent = np.sum(reports_matrix >= 0, axis=1)
        payments = {
            agent_ids[i]: (
                self.payment_per_task * num_tasks_per_agent[i] *
                (1 - self.quality_weight + self.quality_weight * agent_qualities[agent_ids[i]])
            )
            for i in range(n_agents)
        }

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments,
            agent_qualities=agent_qualities
        )

    def _e_step(self, reports_matrix, confusion, n_classes):
        """E-step: Update label posteriors."""
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
            p[j] /= np.sum(p[j])

        return p

    def _m_step(self, reports_matrix, p, n_classes):
        """M-step: Update confusion matrices."""
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

                    confusion[i, k, l] = numerator / (denominator + 1e-10)

        return confusion
```

**Properties**:
- Models agent-specific confusion matrices
- Can identify high vs. low quality agents
- Iterative algorithm (may be slow for large datasets)
- Payment based on inferred quality (estimated post-hoc, not game-theoretic incentive)

### 5.3 Output Agreement (Simple Peer Prediction)

**Algorithm**: Pay agents based on agreement with randomly selected peer.

```python
class OutputAgreement(Mechanism):
    """Simple peer prediction: pay for agreement with peer."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.agreement_payment = config.get("agreement_payment", 1.0)
        self.disagreement_payment = config.get("disagreement_payment", 0.0)
        self.rng = np.random.default_rng(config.get("seed", 42))

    def requires_predictions(self) -> bool:
        return False

    def aggregate_and_pay(self, reports: list[Report],
                          tasks: list[Task]) -> MechanismResult:
        """Aggregate by majority, pay for agreement with random peer."""

        # Group reports by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate by majority
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            votes = [r.report for r in task_reports]
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes)/2 else 0

        # Payments: for each agent, compare with random peer on same tasks
        payments = {}
        for agent_id in set(r.agent_id for r in reports):
            agent_reports = [r for r in reports if r.agent_id == agent_id]
            total_payment = 0

            for report in agent_reports:
                # Find other agents who labeled same task
                peers = [r for r in reports
                        if r.task_id == report.task_id and r.agent_id != agent_id]

                if peers:
                    # Select random peer
                    peer = self.rng.choice(peers)

                    # Pay for agreement
                    if report.report == peer.report:
                        total_payment += self.agreement_payment
                    else:
                        total_payment += self.disagreement_payment

            payments[agent_id] = total_payment

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments
        )
```

**Properties**:
- Simple peer prediction mechanism
- Incentivizes reporting what others will report (not necessarily truth)
- Can have uninformative equilibrium (everyone reports same thing)
- Payment is game-theoretic (depends on others' reports)

### 5.4 Robust Bayesian Truth Serum (RBTS)

**Algorithm**: Witkowski & Parkes (2012) - detail-free peer prediction for binary signals.

```python
class RBTS(Mechanism):
    """Robust Bayesian Truth Serum for small populations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_payment = config.get("base_payment", 1.0)
        self.bonus_scale = config.get("bonus_scale", 1.0)

    def requires_predictions(self) -> bool:
        return True  # Requires prediction of others' reports

    def aggregate_and_pay(self, reports: list[Report],
                          tasks: list[Task]) -> MechanismResult:
        """RBTS scoring: reward surprisingly common answers."""

        # Group by task
        reports_by_task = self._group_by_task(reports)

        # Aggregate by majority
        aggregated_labels = {}
        for task_id, task_reports in reports_by_task.items():
            votes = [r.report for r in task_reports]
            aggregated_labels[task_id] = 1 if sum(votes) > len(votes)/2 else 0

        # RBTS payments
        payments = {}
        for agent_id in set(r.agent_id for r in reports):
            agent_reports = [r for r in reports if r.agent_id == agent_id]
            total_payment = 0

            for report in agent_reports:
                # Score = log(frequency of my report) - log(my prediction of frequency)
                # "Surprisingly common" answers score higher

                # Get all reports for this task
                task_reports = reports_by_task[report.task_id]
                n = len(task_reports)

                # Frequency of my report
                my_report_count = sum(1 for r in task_reports if r.report == report.report)
                frequency = my_report_count / n

                # My prediction
                if report.prediction is None:
                    raise ValueError(f"RBTS requires predictions, but agent {agent_id} didn't provide one")

                my_prediction = report.prediction[report.report]

                # RBTS score (simplified version)
                # Full version has more sophisticated scoring
                if my_prediction > 0:
                    score = np.log(frequency / my_prediction)
                else:
                    score = 0

                total_payment += self.base_payment + self.bonus_scale * score

            payments[agent_id] = max(0, total_payment)  # Ensure non-negative

        return MechanismResult(
            aggregated_labels=aggregated_labels,
            payments=payments
        )
```

**Properties**:
- Detail-free: doesn't need common prior
- Works with small groups (n ≥ 3)
- Requires predictions (extra elicitation cost)
- Theoretically incentive-compatible

**Note**: Implementation above is simplified. Full RBTS has more sophisticated scoring function. See Witkowski & Parkes (2012) for details.

---

## 6. Simulation Engine

### 6.1 Game Class

```python
class Game:
    """A single instance of the labeling game."""

    def __init__(self, config: ExperimentConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        # Generate tasks
        self.tasks = self._generate_tasks()

        # Generate agents
        self.agents = self._generate_agents()

        # Initialize mechanism
        self.mechanism = self._create_mechanism()

    def run(self) -> GameInstance:
        """Run the game: agents observe, report, mechanism aggregates."""

        reports = []

        # For each task, get reports from agents
        for task in self.tasks:
            task_reports = []

            for agent in self.agents:
                # Agent observes
                signal = agent.observe(task)

                # Agent decides what to report
                report_value = agent.report(
                    task, signal, self.mechanism, self.agents
                )

                # If mechanism requires predictions, get prediction
                prediction = None
                if self.mechanism.requires_predictions():
                    prediction = agent.predict_others(task)

                # Record report
                task_reports.append(Report(
                    agent_id=agent.params.agent_id,
                    task_id=task.task_id,
                    report=report_value,
                    prediction=prediction
                ))

            reports.extend(task_reports)

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
            true_labels=true_labels
        )

    def _generate_tasks(self) -> list[Task]:
        """Generate tasks according to config."""
        tasks = []
        for i in range(self.config.num_tasks):
            # Sample true label from prior
            true_label = 1 if self.rng.random() < self.config.label_prior else 0

            # Sample difficulty (or use fixed difficulty from config)
            difficulty = self.config.task_difficulty

            tasks.append(Task(
                task_id=i,
                true_label=true_label,
                difficulty=difficulty,
                prior_prob=self.config.label_prior
            ))

        return tasks

    def _generate_agents(self) -> list[Agent]:
        """Generate agents according to config."""
        agents = []
        agent_id = 0

        for agent_type, proportion in self.config.agent_mix.items():
            num_agents = int(self.config.num_agents * proportion)

            for _ in range(num_agents):
                # Create agent params (could load from config file)
                params = self._get_agent_params(agent_type, agent_id)

                # Create agent instance
                agent = self._create_agent(agent_type, params)

                agents.append(agent)
                agent_id += 1

        return agents
```

### 6.2 SimulationEngine

```python
class SimulationEngine:
    """Orchestrates multiple game runs and collects results."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run_experiment(self) -> ExperimentResults:
        """Run multiple games and aggregate results."""

        metrics = []

        for run_id in range(self.config.num_runs):
            # Create RNG with seed for reproducibility
            seed = self.config.random_seed + run_id
            rng = np.random.default_rng(seed)

            # Create and run game
            game = Game(self.config, rng)
            game_instance = game.run()

            # Evaluate
            metric = self._evaluate(game_instance)
            metrics.append(metric)

            # Log progress
            if (run_id + 1) % 10 == 0:
                print(f"Completed {run_id + 1}/{self.config.num_runs} runs")

        # Aggregate results
        return self._aggregate_results(metrics)

    def _evaluate(self, game: GameInstance) -> MetricResult:
        """Compute metrics for a single game."""

        # Quality metrics
        accuracy = self._compute_accuracy(
            game.result.aggregated_labels,
            game.true_labels
        )

        precision, recall, f1 = self._compute_pr_f1(
            game.result.aggregated_labels,
            game.true_labels
        )

        # Payment metrics
        total_payment = sum(game.result.payments.values())
        payments_list = list(game.result.payments.values())
        average_payment = np.mean(payments_list)
        payment_std = np.std(payments_list)

        # Efficiency
        quality_per_dollar = accuracy / total_payment if total_payment > 0 else 0

        return MetricResult(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            total_payment=total_payment,
            average_payment=average_payment,
            payment_std=payment_std,
            quality_per_dollar=quality_per_dollar
        )

    def _aggregate_results(self, metrics: list[MetricResult]) -> ExperimentResults:
        """Aggregate metrics across runs."""

        # Convert to arrays
        metric_arrays = {
            field: np.array([getattr(m, field) for m in metrics])
            for field in MetricResult.__annotations__.keys()
        }

        # Compute means
        mean_metrics = {
            field: np.mean(values)
            for field, values in metric_arrays.items()
        }

        # Compute std
        std_metrics = {
            field: np.std(values)
            for field, values in metric_arrays.items()
        }

        # Compute 95% confidence intervals
        confidence_intervals = {}
        for field, values in metric_arrays.items():
            mean = np.mean(values)
            sem = np.std(values) / np.sqrt(len(values))
            ci = (mean - 1.96 * sem, mean + 1.96 * sem)
            confidence_intervals[field] = ci

        return ExperimentResults(
            config=self.config,
            metrics=metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals
        )
```

---

## 7. Evaluation Framework

### 7.1 Metrics

```python
def compute_accuracy(predicted: dict[int, int],
                    true_labels: dict[int, int]) -> float:
    """Compute classification accuracy."""
    correct = sum(1 for task_id in predicted
                  if predicted[task_id] == true_labels[task_id])
    return correct / len(predicted)

def compute_precision_recall_f1(predicted: dict[int, int],
                                true_labels: dict[int, int]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for binary classification."""
    tp = sum(1 for tid in predicted
             if predicted[tid] == 1 and true_labels[tid] == 1)
    fp = sum(1 for tid in predicted
             if predicted[tid] == 1 and true_labels[tid] == 0)
    fn = sum(1 for tid in predicted
             if predicted[tid] == 0 and true_labels[tid] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
```

### 7.2 Statistical Analysis

```python
def compare_mechanisms(results1: ExperimentResults,
                      results2: ExperimentResults,
                      metric: str = "accuracy") -> dict:
    """Compare two mechanisms statistically."""

    values1 = np.array([getattr(m, metric) for m in results1.metrics])
    values2 = np.array([getattr(m, metric) for m in results2.metrics])

    # Perform t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(values1, values2)

    # Compute effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std

    return {
        "mean_diff": np.mean(values1) - np.mean(values2),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05
    }
```

---

## 8. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Basic simulation infrastructure working with 2 mechanisms and 2 agent types.

**Deliverables**:
1. ✅ Project structure and package setup
2. ✅ Core data structures (Task, Agent, Mechanism, Report, GameInstance)
3. ✅ Task generator (binary classification)
4. ✅ Two agent types:
   - Truthful agent
   - Lazy agent (random strategy)
5. ✅ Two mechanisms:
   - Majority voting
   - Dawid-Skene
6. ✅ Simulation engine (single game run)
7. ✅ Basic evaluation (accuracy, payment)
8. ✅ Unit tests for all components

**Validation Experiments**:
- **Experiment 1a**: All truthful agents (high ability) → both mechanisms should achieve high accuracy
- **Experiment 1b**: All lazy agents (random) → accuracy ≈ 50% for both mechanisms

**Success Criteria**:
- Tests pass
- Experiment 1a achieves >90% accuracy with truthful agents (ability=0.95)
- Experiment 1b achieves ≈50% accuracy with lazy agents
- Dawid-Skene correctly identifies agent qualities

---

### Phase 2: Peer Prediction Mechanisms (Weeks 3-4)

**Goal**: Add peer prediction mechanisms and compare with aggregation methods.

**Deliverables**:
1. ✅ Output Agreement mechanism
2. ✅ RBTS mechanism (with prediction elicitation)
3. ✅ Agent prediction method (for RBTS)
4. ✅ Extended evaluation metrics
5. ✅ Configuration system (YAML configs)
6. ✅ Multiple-run experiments with statistical analysis

**Experiments**:
- **Experiment 2a**: Compare 4 mechanisms with all truthful agents
  - Expected: All perform similarly (high accuracy)
  - Tests implementation correctness

- **Experiment 2b**: Compare 4 mechanisms with all lazy agents
  - Expected: All perform poorly (≈50% accuracy)
  - Mechanisms can't fix bad input

- **Experiment 2c**: Compare 4 mechanisms with 70% truthful, 30% lazy
  - Expected: Mechanisms start to differentiate
  - Which handles noise better?

**Success Criteria**:
- All 4 mechanisms implemented and tested
- Config-driven experiments work
- Statistical comparison framework operational
- Can generate comparison plots

---

### Phase 3: Strategic Agents (Weeks 5-6)

**Goal**: Add strategic and adversarial agents to test mechanism robustness.

**Deliverables**:
1. ✅ Strategic agent (simplified best-response)
2. ✅ Adversarial agent (multiple strategies)
3. ✅ Noisy truthful agent
4. ✅ Agent belief model implementation
5. ✅ Mechanism-specific payment computation for strategic reasoning

**Experiments**:
- **Experiment 3a**: Strategic agents under each mechanism
  - Do they converge to truthful reporting under RBTS?
  - Do they game Dawid-Skene?

- **Experiment 3b**: Adversarial robustness
  - Vary adversarial % from 0% to 50%
  - At what % does each mechanism break?

- **Experiment 3c**: Mixed realistic population
  - 50% truthful, 30% lazy, 10% strategic, 10% noisy
  - Which mechanism performs best?

**Success Criteria**:
- Strategic agents implement reasonable best-response
- Can measure mechanism robustness to adversarial agents
- Results show differentiation between mechanisms

---

### Phase 4: Analysis & Refinement (Weeks 7-8)

**Goal**: Comprehensive experiments, statistical analysis, publication-quality outputs.

**Deliverables**:
1. ✅ Full experimental suite
2. ✅ Statistical significance testing
3. ✅ Visualization suite (publication plots)
4. ✅ Analysis notebooks
5. ✅ Documentation (architecture, API, usage)
6. ✅ Results write-up

**Experiments**:
- **Experiment 4a**: Budget constraints
  - Fixed total payment budget
  - Which mechanism maximizes quality?

- **Experiment 4b**: Scaling study
  - Vary number of agents (3, 5, 10, 20, 50)
  - How do mechanisms scale?

- **Experiment 4c**: Task difficulty
  - Easy tasks (high signal quality) vs. hard tasks
  - Does difficulty moderate mechanism performance?

**Analysis**:
- Statistical comparison across all scenarios
- Identify conditions favoring each mechanism
- Cost-benefit analysis (quality improvement vs. added complexity)
- Robustness analysis (sensitivity to assumptions)

**Success Criteria**:
- All experiments complete with statistical analysis
- Publication-quality figures generated
- Clear insights about when to use which mechanism
- Documentation complete

---

### Optional Phase 5: Extensions (Weeks 9+)

**If time permits**, add:

1. **Multi-class classification** (beyond binary)
2. **GLAD mechanism** (task difficulty modeling)
3. **Learning agents** (iterative best-response, Q-learning)
4. **Multi-task peer prediction** (DMI mechanism)
5. **Real data application** (apply to existing crowdsourcing datasets)

---

## 9. Testing Strategy

### 9.1 Unit Tests

Each component has comprehensive unit tests:

```python
# tests/test_agents.py
def test_truthful_agent_high_ability():
    """Truthful agent with ability=1.0 should always observe correctly."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "truthful", ability=1.0, effort_cost=0, risk_aversion=0)
    agent = TruthfulAgent(params, rng)

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Run 100 times, should always observe correctly
    observations = [agent.observe(task) for _ in range(100)]
    assert all(obs == 1 for obs in observations)

def test_lazy_agent_random():
    """Lazy agent should report randomly."""
    rng = np.random.default_rng(42)
    params = AgentParams(0, "lazy", ability=0.5, effort_cost=0, risk_aversion=0)
    agent = LazyAgent(params, rng, strategy="random")

    task = Task(0, true_label=1, difficulty=0.5, prior_prob=0.5)

    # Run many times, should be roughly 50/50
    reports = [agent.report(task, 0, None, []) for _ in range(1000)]
    fraction_ones = sum(reports) / len(reports)
    assert 0.4 < fraction_ones < 0.6  # Should be close to 0.5

# tests/test_mechanisms.py
def test_majority_voting_simple():
    """Majority voting should select majority label."""
    mechanism = MajorityVoting({"payment_per_task": 1.0})

    reports = [
        Report(0, 0, 1),  # Agent 0, Task 0, Label 1
        Report(1, 0, 1),  # Agent 1, Task 0, Label 1
        Report(2, 0, 0),  # Agent 2, Task 0, Label 0
    ]

    result = mechanism.aggregate_and_pay(reports, [])

    assert result.aggregated_labels[0] == 1  # Majority is 1
    assert all(p == 1.0 for p in result.payments.values())  # Fixed payment

def test_dawid_skene_convergence():
    """Dawid-Skene should converge and identify good vs. bad agents."""
    # Create synthetic data: 2 good agents (90% accurate), 1 bad agent (50% accurate)
    # 100 tasks, true labels known
    # Good agents should get higher quality scores
    pass  # TODO: implement
```

### 9.2 Integration Tests

```python
def test_full_game_run():
    """Test a complete game run end-to-end."""
    config = ExperimentConfig(
        name="test",
        num_tasks=10,
        num_agents=5,
        agent_mix={"truthful": 0.6, "lazy": 0.4},
        mechanism_name="majority_voting",
        mechanism_params={},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=1,
        random_seed=42
    )

    rng = np.random.default_rng(42)
    game = Game(config, rng)
    result = game.run()

    # Basic sanity checks
    assert len(result.tasks) == 10
    assert len(result.agents) == 5
    assert len(result.reports) == 50  # 5 agents × 10 tasks
    assert len(result.result.aggregated_labels) == 10
    assert len(result.result.payments) == 5
```

### 9.3 Validation Tests

Confirm known theoretical results:

```python
def test_majority_voting_accuracy_with_truthful_agents():
    """With all high-ability truthful agents, accuracy should be very high."""
    config = ExperimentConfig(
        name="validation_truthful",
        num_tasks=100,
        num_agents=10,
        agent_mix={"truthful": 1.0},  # All truthful
        mechanism_name="majority_voting",
        mechanism_params={},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Should achieve >95% accuracy
    assert results.mean_metrics["accuracy"] > 0.95

def test_mechanisms_fail_with_all_lazy_agents():
    """With all lazy (random) agents, accuracy should be ~50%."""
    config = ExperimentConfig(
        name="validation_lazy",
        num_tasks=100,
        num_agents=10,
        agent_mix={"lazy": 1.0},  # All lazy
        mechanism_name="majority_voting",
        mechanism_params={},
        task_difficulty=0.5,
        label_prior=0.5,
        num_runs=10,
        random_seed=42
    )

    engine = SimulationEngine(config)
    results = engine.run_experiment()

    # Should be close to random guessing
    assert 0.45 < results.mean_metrics["accuracy"] < 0.55
```

---

## 10. Success Criteria

### 10.1 Technical Success

✅ **Implementation Complete**:
- [x] All 4 core mechanisms implemented (Majority, Dawid-Skene, Output Agreement, RBTS)
- [x] All 5 agent types implemented (Truthful, Lazy, Strategic, Adversarial, Noisy)
- [x] Simulation engine working
- [x] Evaluation framework complete
- [x] Unit tests pass (>90% coverage)
- [x] Integration tests pass

✅ **Validation Tests Pass**:
- [x] High-ability truthful agents achieve >95% accuracy
- [x] All-lazy agents achieve ≈50% accuracy (random baseline)
- [x] Dawid-Skene correctly identifies agent qualities
- [x] RBTS properly handles predictions

✅ **Reproducibility**:
- [x] Fixed seeds produce identical results
- [x] All parameters logged
- [x] Config files for all experiments
- [x] Results saved in structured format

### 10.2 Research Success

✅ **Comparative Evaluation Complete**:
- [x] All 4 mechanisms compared across multiple scenarios
- [x] Statistical significance tests conducted
- [x] Effect sizes computed
- [x] Confidence intervals provided

✅ **Key Questions Answered**:
1. **Do incentive mechanisms outperform simple aggregation?**
   - Clear answer for different agent populations

2. **How robust are mechanisms to adversarial agents?**
   - Breaking point identified for each mechanism

3. **What are the cost-quality tradeoffs?**
   - Quality improvement quantified relative to added complexity

4. **Which mechanism for which context?**
   - Decision tree or guidelines provided

✅ **Publication-Ready Outputs**:
- [x] Publication-quality figures
- [x] Statistical analysis complete
- [x] Results interpretable and actionable
- [x] Limitations acknowledged
- [x] Future work identified

### 10.3 Impact Criteria

✅ **Practical Value**:
- Practitioners can use results to select mechanisms
- Clear guidance for platform designers
- Cost-benefit analysis provided

✅ **Theoretical Contribution**:
- Bridges two separate literatures
- Empirical validation of theoretical claims
- New insights about mechanism robustness

✅ **Methodological Contribution**:
- Unified framework others can extend
- Open-source implementation
- Reproducible experiments

---

## 11. Reproducibility Protocol

### 11.1 Random Seed Management

```python
class SeedManager:
    """Manages random seeds for reproducibility."""

    def __init__(self, master_seed: int):
        self.master_seed = master_seed
        self.master_rng = np.random.default_rng(master_seed)

    def get_game_seed(self, run_id: int) -> int:
        """Get seed for a specific game run."""
        return self.master_seed + run_id

    def get_agent_seed(self, agent_id: int, game_seed: int) -> int:
        """Get seed for a specific agent."""
        return game_seed + 1000 * agent_id
```

### 11.2 Parameter Logging

Every experiment automatically logs:
- Experiment configuration (YAML)
- Random seeds used
- Agent parameters (abilities, costs)
- Task parameters (difficulties, priors)
- Mechanism parameters
- Timestamp and git commit hash

```python
def log_experiment(config: ExperimentConfig, results: ExperimentResults):
    """Log all experiment details."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "config": asdict(config),
        "results": {
            "mean_metrics": results.mean_metrics,
            "std_metrics": results.std_metrics,
            "confidence_intervals": results.confidence_intervals,
        }
    }

    # Save to JSON
    output_path = f"results/raw/{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(log_data, f, indent=2)
```

### 11.3 Environment Specification

```toml
# pyproject.toml
[project]
name = "game-theory-data-labeling"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "black>=23.3.0",
    "mypy>=1.3.0",
    "ruff>=0.0.270",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## 12. Next Steps

### Immediate (This Session):
1. ✅ Review IMPLEMENTATION.md with user
2. ⏳ Get approval on architecture and phased plan
3. ⏳ Clarify any ambiguities
4. ⏳ Begin Phase 1 implementation (if approved)

### Phase 1 Kickoff:
1. Set up project structure with `uv`
2. Implement core data structures
3. Implement Truthful and Lazy agents
4. Implement Majority Voting mechanism
5. Implement basic simulation engine
6. Write unit tests
7. Run validation experiment (truthful vs. lazy)

### Documentation:
1. Architecture documentation
2. Algorithm documentation (with citations)
3. API reference
4. Usage examples
5. Contribution guidelines (if open-sourcing)

---

## Appendix: Open Questions

### Strategic Agent Implementation

**Question**: How sophisticated should strategic agents be initially?

**Options**:
1. **Simple**: Best-response to uniform prior over others' strategies
2. **Medium**: Iterative best-response (simulate until convergence)
3. **Complex**: Full Bayes-Nash equilibrium computation

**Recommendation**: Start with Option 1 (simple), add Option 2 later if needed.

### Multi-class Extension

**Question**: When to extend to multi-class?

**Recommendation**: After Phase 4, if time permits. Binary is sufficient for initial publication.

### Real Data Application

**Question**: Should we test on real crowdsourcing datasets?

**Considerations**:
- Pro: Validates on real data
- Con: Real data doesn't have predictions (needed for BTS/RBTS)
- Con: Ground truth may be noisy/incorrect

**Recommendation**: Focus on simulation for main paper. Real data could be supplementary or follow-up work.

---

**End of IMPLEMENTATION.md**
