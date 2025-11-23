# Implementation Plan: Labor-for-Content-Access Prototype

## Document Overview

This document provides the detailed technical design for implementing a research prototype that explores data labeling as an alternative content monetization model. The system simulates a marketplace where users choose between viewing ads, paying subscriptions, or performing data labeling tasks to access creator content.

**Last Updated**: 2025-11-23
**Status**: Design Phase
**Target Timeline**: 6-8 weeks

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Data Schemas](#3-data-schemas)
4. [Core Components](#4-core-components)
5. [Implementation Phases](#5-implementation-phases)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Quality Metrics](#7-quality-metrics)
8. [Success Criteria](#8-success-criteria)
9. [Testing Strategy](#9-testing-strategy)
10. [Reproducibility](#10-reproducibility)

---

## 1. Architecture Overview

### 1.1 High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     Research Prototype System                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌──────────────────────────────────┐
│   Data Layer        │     │   Simulation Layer               │
│                     │     │                                  │
│  ┌──────────────┐   │     │  ┌────────────┐  ┌────────────┐ │
│  │ MNIST        │   │────▶│  │ User       │  │ Creator    │ │
│  │ CIFAR-10     │   │     │  │ Agents     │  │ Agents     │ │
│  │ SST-2        │   │     │  └────────────┘  └────────────┘ │
│  └──────────────┘   │     │                                  │
│                     │     │  ┌────────────┐  ┌────────────┐ │
│  ┌──────────────┐   │     │  │ Company    │  │ Marketplace│ │
│  │ Ground Truth │   │────▶│  │ Agents     │  │ Engine     │ │
│  │ Labels       │   │     │  └────────────┘  └────────────┘ │
│  └──────────────┘   │     └──────────────────────────────────┘
└─────────────────────┘                  │
                                         │
┌─────────────────────┐                  │
│   Task Layer        │                  │
│                     │                  ▼
│  ┌──────────────┐   │     ┌──────────────────────────────────┐
│  │ Task         │   │────▶│   Quality Control Layer          │
│  │ Generator    │   │     │                                  │
│  └──────────────┘   │     │  ┌────────────┐  ┌────────────┐ │
│                     │     │  │ Consensus  │  │ Validation │ │
│  ┌──────────────┐   │     │  │ Algorithm  │  │ Engine     │ │
│  │ Task Router  │   │────▶│  └────────────┘  └────────────┘ │
│  └──────────────┘   │     └──────────────────────────────────┘
└─────────────────────┘                  │
                                         │
┌─────────────────────┐                  │
│   Revenue Layer     │                  │
│                     │                  ▼
│  ┌──────────────┐   │     ┌──────────────────────────────────┐
│  │ Pricing      │   │────▶│   Analytics & Reporting          │
│  │ Engine       │   │     │                                  │
│  └──────────────┘   │     │  ┌────────────┐  ┌────────────┐ │
│                     │     │  │ Metrics    │  │ Visualize  │ │
│  ┌──────────────┐   │     │  │ Calculator │  │ Results    │ │
│  │ Distribution │   │────▶│  └────────────┘  └────────────┘ │
│  │ Logic        │   │     └──────────────────────────────────┘
│  └──────────────┘   │
└─────────────────────┘
```

### 1.2 System Components

**Core Modules**:
1. **Data Management**: Load and manage datasets (MNIST, CIFAR-10, SST-2)
2. **Agent Simulation**: Simulate users, creators, and ML companies
3. **Task Management**: Generate, route, and track labeling tasks
4. **Quality Control**: Validate labels, compute consensus, assess quality
5. **Revenue Engine**: Calculate pricing, distribute revenue
6. **Analytics**: Collect metrics, generate reports, visualize results

**Supporting Modules**:
- Configuration management
- Logging and monitoring
- Random seed management (reproducibility)
- Data persistence (SQLite or JSON)

### 1.3 Data Flow

```
1. Initialize System
   ├─ Load datasets (MNIST, CIFAR-10, SST-2)
   ├─ Create agent populations (users, creators, companies)
   └─ Configure marketplace parameters

2. Simulation Loop (per content access event)
   ├─ User agent visits creator's content
   ├─ System presents choice: [Ads | Payment | Labor]
   ├─ User agent makes decision based on utility function
   │
   ├─ IF user chooses Labor:
   │  ├─ Marketplace routes N tasks to user
   │  ├─ User completes tasks (simulated labeling)
   │  ├─ System validates labels
   │  ├─ Quality control checks consensus
   │  ├─ Revenue calculated and distributed
   │  └─ User granted content access
   │
   ├─ ELSE IF user chooses Ads:
   │  ├─ Simulate ad viewing time
   │  ├─ Calculate ad revenue (CPM)
   │  └─ User granted content access
   │
   └─ ELSE (user chooses Payment):
      ├─ Charge payment
      └─ User granted content access

3. Collect Metrics
   ├─ Choice distribution (% Labor vs Ads vs Payment)
   ├─ Label quality (accuracy, Fleiss' kappa)
   ├─ Revenue comparison (Labor vs Ads)
   ├─ User satisfaction proxies
   └─ Economic viability metrics

4. Generate Reports
   ├─ Summary statistics
   ├─ Visualizations
   └─ Research findings
```

---

## 2. Technology Stack

### 2.1 Core Dependencies

```python
# pyproject.toml or requirements.txt

# Core ML/Data
datasets >= 2.14.0        # Hugging Face datasets
torch >= 2.0.0            # PyTorch (if needed for models)
numpy >= 1.24.0           # Numerical computing
pandas >= 2.0.0           # Data manipulation

# Agent-Based Modeling
mesa >= 2.1.0             # Agent-based modeling framework
simpy >= 4.0.0            # Discrete event simulation (optional)

# Quality Metrics
scikit-learn >= 1.3.0     # ML utilities, kappa metrics
statsmodels >= 0.14.0     # Statistical models, inter-rater agreement
# crowd-kit >= 1.0.0      # Crowdsourcing quality control (optional)

# Visualization
matplotlib >= 3.7.0       # Plotting
seaborn >= 0.12.0         # Statistical visualizations
plotly >= 5.14.0          # Interactive plots (optional)

# Utilities
pydantic >= 2.0.0         # Data validation
tqdm >= 4.65.0            # Progress bars
python-dotenv >= 1.0.0    # Environment configuration

# Development
pytest >= 7.3.0           # Testing
black >= 23.0.0           # Code formatting
mypy >= 1.3.0             # Type checking
```

### 2.2 Python Version

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10 or 3.11
- **Package Manager**: `uv` (as specified in CLAUDE.md)

### 2.3 Installation

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

---

## 3. Data Schemas

### 3.1 Task Schema

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_SENTIMENT = "text_sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    AUDIO_EMOTION = "audio_emotion"

class TaskDifficulty(str, Enum):
    EASY = "easy"      # MNIST-level
    MEDIUM = "medium"  # CIFAR-10, SST-2
    HARD = "hard"      # NER, complex images

class Task(BaseModel):
    """Represents a single labeling task"""

    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType
    difficulty: TaskDifficulty

    # Data
    dataset_name: str  # e.g., "mnist", "cifar10", "sst2"
    item_index: int    # Index in the dataset
    item_data: Dict[str, Any]  # The actual data (image tensor, text, etc.)

    # Labels
    ground_truth_label: Any  # True label (for validation)
    possible_labels: list[str]  # e.g., ["0", "1", ..., "9"] for MNIST

    # Metadata
    company_id: str  # Which company needs this labeled
    created_at: datetime
    estimated_time_seconds: float  # Expected completion time

    # Pricing
    price_per_label: float  # What company pays per label
    quality_requirement: float  # Minimum Fleiss' kappa required

class Label(BaseModel):
    """Represents a user's label submission"""

    label_id: str
    task_id: str
    user_id: str

    # Submission
    submitted_label: Any  # User's answer
    confidence: Optional[float] = None  # User's confidence (if captured)
    time_taken_seconds: float
    submitted_at: datetime

    # Validation
    is_correct: Optional[bool] = None  # Compared to ground truth
    quality_score: Optional[float] = None  # From quality control algorithms
```

### 3.2 User Agent Schema

```python
class UserType(str, Enum):
    TASK_AVOIDER = "task_avoider"
    BALANCED = "balanced"
    TASK_PREFERER = "task_preferer"
    PAYMENT_PREFERER = "payment_preferer"

class UserAgent(BaseModel):
    """Represents a simulated user"""

    user_id: str
    user_type: UserType

    # Behavioral parameters
    time_sensitivity: Literal["low", "medium", "high"]
    task_difficulty_tolerance: Literal["easy_only", "medium", "all"]
    ad_tolerance: float = Field(ge=0, le=1)  # 0 = hates ads, 1 = doesn't mind
    privacy_concern: Literal["low", "medium", "high"]

    # Decision weights (for utility function)
    weight_time: float = 1.0
    weight_difficulty: float = 1.0
    weight_ads: float = 1.0
    weight_payment: float = 1.0

    # Performance tracking
    tasks_completed: int = 0
    total_time_spent: float = 0.0
    accuracy_rate: float = 0.0  # Running average

    # Labeling ability (simulated skill level)
    base_accuracy: float = Field(ge=0, le=1, default=0.85)  # Base accuracy on easy tasks

    # Note: difficulty_penalty should be set in __init__ or use Field(default_factory=...)
    # to avoid mutable default issues
    difficulty_penalty: Optional[Dict[TaskDifficulty, float]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.difficulty_penalty is None:
            self.difficulty_penalty = {
                TaskDifficulty.EASY: 0.0,
                TaskDifficulty.MEDIUM: 0.1,
                TaskDifficulty.HARD: 0.2
            }

    def make_choice(self, content_value: float, ad_time: float,
                    payment_cost: float, task_count: int,
                    task_difficulty: TaskDifficulty) -> Literal["ads", "payment", "labor"]:
        """
        Decision function using utility maximization

        U(choice) = value(content) - cost(choice)
        """
        # Implement utility calculation (to be detailed in code)
        pass

    def simulate_labeling(self, task: Task) -> Label:
        """
        Simulate user labeling a task
        Returns label with accuracy based on user's skill and task difficulty
        """
        pass
```

### 3.3 Creator Agent Schema

```python
class CreatorSize(str, Enum):
    SMALL = "small"      # 1K-10K monthly visitors
    MEDIUM = "medium"    # 10K-100K monthly visitors
    LARGE = "large"      # 100K+ monthly visitors

class CreatorAgent(BaseModel):
    """Represents a content creator/website"""

    creator_id: str
    size: CreatorSize

    # Traffic characteristics
    monthly_visitors: int
    avg_visitors_per_day: int
    visitor_distribution: str = "uniform"  # or "poisson", "normal"

    # Current monetization
    current_cpm: float  # Current ad CPM (e.g., $3.12 - $8.60)
    current_monthly_ad_revenue: float

    # Content characteristics
    content_value: float = Field(ge=0, le=10)  # How valuable is content (affects user choice)
    content_type: str = "article"  # article, video, tool, etc.

    # Adoption of labor-for-access
    offers_labor_option: bool = True
    labor_task_count: int = 5  # How many tasks user must complete

    # Tracking
    total_access_events: int = 0
    labor_access_count: int = 0
    ad_access_count: int = 0
    payment_access_count: int = 0
    total_labor_revenue: float = 0.0
    total_ad_revenue: float = 0.0
    total_payment_revenue: float = 0.0
```

### 3.4 Company Agent Schema

```python
class CompanySize(str, Enum):
    STARTUP = "startup"
    MIDSIZE = "midsize"
    ENTERPRISE = "enterprise"

class CompanyAgent(BaseModel):
    """Represents an ML company needing labeled data"""

    company_id: str
    size: CompanySize

    # Data needs
    task_types_needed: list[TaskType]
    monthly_label_budget: float  # How much they'll pay per month
    labels_needed_per_month: int

    # Quality requirements
    min_quality_kappa: float = 0.70  # Minimum Fleiss' kappa
    redundancy_factor: int = 3  # How many labels per item (for consensus)

    # Pricing
    max_price_per_label: Dict[TaskDifficulty, float] = {
        TaskDifficulty.EASY: 0.05,
        TaskDifficulty.MEDIUM: 0.15,
        TaskDifficulty.HARD: 0.50
    }
    quality_bonus_multiplier: float = 1.5  # Pay 1.5x for >0.75 kappa

    # Inventory
    available_tasks: list[str] = []  # List of task_ids
    completed_tasks: list[str] = []

    # Tracking
    total_labels_purchased: int = 0
    total_spend: float = 0.0
    avg_quality_received: float = 0.0
```

### 3.5 Session Schema

```python
class ContentAccessSession(BaseModel):
    """Represents a single content access event"""

    session_id: str
    timestamp: datetime

    # Participants
    user_id: str
    creator_id: str

    # User choice
    choice: Literal["ads", "payment", "labor"]
    decision_time_ms: float  # Time to make choice

    # If labor chosen
    tasks_assigned: Optional[list[str]] = None
    tasks_completed: Optional[list[str]] = None
    labels_submitted: Optional[list[str]] = None
    total_task_time_seconds: Optional[float] = None
    avg_label_quality: Optional[float] = None

    # Revenue
    revenue_to_creator: float = 0.0
    revenue_to_platform: float = 0.0
    revenue_source: Optional[str] = None  # "labor", "ads", "payment"

    # Outcome
    access_granted: bool
    user_satisfaction_proxy: Optional[float] = None  # Simulated satisfaction score
```

---

## 4. Core Components

### 4.1 Data Management Module

**File**: `src/data/dataset_manager.py`

```python
class DatasetManager:
    """
    Manages loading and sampling from datasets
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets = {}
        self.ground_truth = {}

    def load_datasets(self):
        """Load all configured datasets from Hugging Face"""
        # Load MNIST
        self.datasets['mnist'] = load_dataset("mnist")

        # Load CIFAR-10
        self.datasets['cifar10'] = load_dataset("cifar10")

        # Load SST-2
        self.datasets['sst2'] = load_dataset("stanfordnlp/sst2")

        # Store ground truth labels
        for name, ds in self.datasets.items():
            self.ground_truth[name] = {
                'train': ds['train']['label'],
                'test': ds['test']['label']
            }

    def sample_task(self, dataset_name: str, task_type: TaskType,
                    split: str = "test") -> Task:
        """
        Sample a random item from dataset and create a Task
        """
        pass

    def get_item(self, dataset_name: str, index: int,
                 split: str = "test") -> Dict[str, Any]:
        """
        Get specific item from dataset
        """
        pass
```

### 4.2 Agent Simulation Module

**File**: `src/agents/user_agent.py`

```python
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class UserAgentMesa(Agent):
    """
    Mesa-based user agent for agent-based modeling
    """

    def __init__(self, unique_id, model, user_type: UserType, **params):
        super().__init__(unique_id, model)
        self.user_data = UserAgent(
            user_id=str(unique_id),
            user_type=user_type,
            **params
        )

    def step(self):
        """
        Execute one step of agent behavior
        In our model: encounter content access decision
        """
        # Select a random creator from model
        creator = self.model.random.choice(self.model.creators)

        # Make access choice decision
        choice = self.decide_access_method(creator)

        # Execute chosen method
        if choice == "labor":
            self.complete_labeling_tasks(creator)
        elif choice == "ads":
            self.view_ads(creator)
        else:  # payment
            self.pay_subscription(creator)

    def decide_access_method(self, creator: CreatorAgent) -> str:
        """
        Utility-based decision making

        Utility function:
        U(ads) = content_value - (ad_time * ad_tolerance_factor)
        U(labor) = content_value - (task_time * difficulty_sensitivity)
        U(payment) = content_value - (price * payment_sensitivity)

        Choose option with highest utility
        """
        content_value = creator.content_value

        # Calculate utilities
        u_ads = content_value - (30 * (1 - self.user_data.ad_tolerance))  # 30 sec ads

        avg_task_time = creator.labor_task_count * 15  # 15 sec per task
        difficulty_factor = 1.0  # Simplified, would check actual task difficulty
        u_labor = content_value - (avg_task_time * difficulty_factor * self.user_data.weight_time / 60)

        payment_cost = 5.0  # $5 typical payment
        u_payment = content_value - (payment_cost * self.user_data.weight_payment)

        # Choose max utility
        utilities = {"ads": u_ads, "labor": u_labor, "payment": u_payment}
        return max(utilities, key=utilities.get)

    def complete_labeling_tasks(self, creator: CreatorAgent):
        """
        Simulate completing labeling tasks
        """
        # Get tasks from marketplace
        tasks = self.model.marketplace.assign_tasks(
            user_id=self.user_data.user_id,
            count=creator.labor_task_count
        )

        # Complete each task
        labels = []
        for task in tasks:
            label = self.simulate_label(task)
            labels.append(label)

        # Submit to quality control
        revenue = self.model.quality_control.process_labels(labels)

        # Distribute revenue
        self.model.revenue_engine.distribute(revenue, creator.creator_id)
```

**File**: `src/agents/creator_agent.py`

```python
class CreatorAgentMesa(Agent):
    """Mesa-based creator agent"""

    def __init__(self, unique_id, model, size: CreatorSize, **params):
        super().__init__(unique_id, model)
        self.creator_data = CreatorAgent(
            creator_id=str(unique_id),
            size=size,
            **params
        )

    def step(self):
        """
        Track daily visitors and revenue
        """
        # Simulated daily visitors
        daily_visitors = self.creator_data.avg_visitors_per_day

        # These visitors will interact with user agents
        # Tracking happens in user agent interactions
        pass
```

**File**: `src/agents/company_agent.py`

```python
class CompanyAgentMesa(Agent):
    """Mesa-based company agent"""

    def __init__(self, unique_id, model, size: CompanySize, **params):
        super().__init__(unique_id, model)
        self.company_data = CompanyAgent(
            company_id=str(unique_id),
            size=size,
            **params
        )

    def step(self):
        """
        Generate labeling tasks for the marketplace
        """
        # Determine how many labels needed this step
        labels_per_step = self.company_data.labels_needed_per_month / 30  # Daily

        # Create tasks
        for _ in range(int(labels_per_step)):
            task = self.create_task()
            self.model.marketplace.add_task(task)

    def create_task(self) -> Task:
        """
        Create a labeling task from this company's needs
        """
        # Sample from task types needed
        task_type = self.model.random.choice(self.company_data.task_types_needed)

        # Generate task from dataset
        task = self.model.dataset_manager.sample_task(
            dataset_name=self.get_dataset_for_type(task_type),
            task_type=task_type
        )

        task.company_id = self.company_data.company_id
        task.price_per_label = self.get_price_for_difficulty(task.difficulty)

        return task
```

### 4.3 Marketplace Engine

**File**: `src/marketplace/marketplace.py`

```python
class MarketplaceEngine:
    """
    Manages task inventory and routing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_inventory: Dict[str, Task] = {}
        self.task_queue: list[str] = []  # Queue of available task_ids

    def add_task(self, task: Task):
        """Company adds task to marketplace"""
        self.task_inventory[task.task_id] = task
        self.task_queue.append(task.task_id)

    def assign_tasks(self, user_id: str, count: int) -> list[Task]:
        """
        Assign N tasks to a user

        Routing strategy:
        - Consider user's skill level (if tracked)
        - Consider task difficulty tolerance
        - Balance load across companies
        - Prioritize high-value tasks
        """
        assigned_tasks = []

        for _ in range(count):
            if not self.task_queue:
                break

            # Simple FIFO for MVP
            # Could implement sophisticated routing later
            task_id = self.task_queue.pop(0)
            task = self.task_inventory[task_id]
            assigned_tasks.append(task)

        return assigned_tasks

    def get_task_statistics(self) -> Dict[str, Any]:
        """Return marketplace statistics"""
        return {
            "total_tasks": len(self.task_inventory),
            "available_tasks": len(self.task_queue),
            "tasks_by_type": self._count_by_type(),
            "tasks_by_difficulty": self._count_by_difficulty(),
            "avg_price": self._average_price()
        }
```

### 4.4 Quality Control System

**File**: `src/quality/quality_control.py`

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from typing import Dict, Any, List

class QualityControlSystem:
    """
    Validates labels and computes quality metrics
    """

    def __init__(self, config: Dict[str, Any], marketplace=None):
        self.config = config
        self.redundancy_factor = config.get("redundancy_factor", 3)
        self.marketplace = marketplace  # Reference to get tasks

        # Storage for multi-label consensus
        self.task_labels: Dict[str, list[Label]] = {}
        self.tasks: Dict[str, Task] = {}  # Cache of tasks for lookup

    def get_task(self, task_id: str) -> Task:
        """Retrieve task by ID from cache or marketplace"""
        if task_id in self.tasks:
            return self.tasks[task_id]
        elif self.marketplace:
            task = self.marketplace.task_inventory.get(task_id)
            if task:
                self.tasks[task_id] = task
                return task
        raise ValueError(f"Task {task_id} not found")

    def process_labels(self, labels: list[Label]) -> float:
        """
        Process submitted labels and calculate revenue

        Returns: revenue generated from these labels
        """
        total_revenue = 0.0

        for label in labels:
            # Store label for consensus
            if label.task_id not in self.task_labels:
                self.task_labels[label.task_id] = []
            self.task_labels[label.task_id].append(label)

            # Check if we have enough labels for consensus
            if len(self.task_labels[label.task_id]) >= self.redundancy_factor:
                revenue = self.compute_consensus_and_pay(label.task_id)
                total_revenue += revenue

        return total_revenue

    def compute_consensus_and_pay(self, task_id: str) -> float:
        """
        Compute consensus label and pay if quality meets threshold
        """
        labels = self.task_labels[task_id]

        # Get task and ground truth
        task = self.get_task(task_id)

        # Majority vote
        submitted_labels = [l.submitted_label for l in labels]
        consensus_label = Counter(submitted_labels).most_common(1)[0][0]

        # Compute agreement (Fleiss' kappa)
        kappa = self.compute_fleiss_kappa(submitted_labels)

        # Check accuracy against ground truth
        accuracy = 1.0 if consensus_label == task.ground_truth_label else 0.0

        # Determine payment
        if kappa >= task.quality_requirement:
            # Quality meets threshold
            base_payment = task.price_per_label * len(labels)

            # Bonus for exceptional quality
            # Get bonus multiplier from config (default 1.5)
            bonus_multiplier = self.config.get("quality_bonus_multiplier", 1.5)
            if kappa > 0.75:
                payment = base_payment * bonus_multiplier
            else:
                payment = base_payment

            return payment
        else:
            # Quality too low, no payment
            return 0.0

    def compute_fleiss_kappa(self, labels: list[Any]) -> float:
        """
        Compute Fleiss' kappa for multiple annotators

        For simplicity in prototype, using agreement rate as proxy
        Real implementation would use statsmodels.stats.inter_rater
        """
        if len(labels) < 2:
            return 1.0

        # Simple agreement rate
        most_common = Counter(labels).most_common(1)[0][1]
        agreement = most_common / len(labels)

        # Rough kappa approximation (simplified)
        # Real: use statsmodels.stats.inter_rater.fleiss_kappa
        num_categories = len(set(labels))
        expected_agreement = 1.0 / num_categories
        kappa = (agreement - expected_agreement) / (1 - expected_agreement)

        return max(0.0, kappa)

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Return overall quality metrics"""
        all_kappas = []
        all_accuracies = []

        for task_id, labels in self.task_labels.items():
            if len(labels) >= self.redundancy_factor:
                submitted = [l.submitted_label for l in labels]
                kappa = self.compute_fleiss_kappa(submitted)
                all_kappas.append(kappa)

                # Check accuracy
                task = self.get_task(task_id)
                consensus = Counter(submitted).most_common(1)[0][0]
                acc = 1.0 if consensus == task.ground_truth_label else 0.0
                all_accuracies.append(acc)

        return {
            "avg_fleiss_kappa": np.mean(all_kappas) if all_kappas else 0.0,
            "avg_accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "tasks_with_consensus": len(all_kappas),
            "kappa_distribution": {
                "excellent (>0.75)": sum(1 for k in all_kappas if k > 0.75),
                "good (0.60-0.75)": sum(1 for k in all_kappas if 0.60 <= k <= 0.75),
                "fair (0.40-0.60)": sum(1 for k in all_kappas if 0.40 <= k < 0.60),
                "poor (<0.40)": sum(1 for k in all_kappas if k < 0.40)
            }
        }
```

### 4.5 Revenue Engine

**File**: `src/revenue/revenue_engine.py`

```python
class RevenueEngine:
    """
    Calculates and distributes revenue
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Revenue split configuration
        self.platform_share = config.get("platform_share", 0.20)  # 20%
        self.creator_share = config.get("creator_share", 0.70)    # 70%
        self.user_share = config.get("user_share", 0.10)          # 10% (optional)

        # Tracking
        self.total_revenue = 0.0
        self.revenue_by_creator: Dict[str, float] = {}
        self.revenue_by_source: Dict[str, float] = {
            "labor": 0.0,
            "ads": 0.0,
            "payment": 0.0
        }

    def distribute(self, revenue: float, creator_id: str,
                   source: str = "labor", user_id: Optional[str] = None):
        """
        Distribute revenue according to configured splits
        """
        self.total_revenue += revenue
        self.revenue_by_source[source] += revenue

        if source == "labor":
            # Split labor revenue
            platform_cut = revenue * self.platform_share
            creator_cut = revenue * self.creator_share
            user_cut = revenue * self.user_share

            # Record
            if creator_id not in self.revenue_by_creator:
                self.revenue_by_creator[creator_id] = 0.0
            self.revenue_by_creator[creator_id] += creator_cut

        elif source == "ads":
            # All ad revenue to creator (platform already took cut)
            if creator_id not in self.revenue_by_creator:
                self.revenue_by_creator[creator_id] = 0.0
            self.revenue_by_creator[creator_id] += revenue

        else:  # payment
            # Payment revenue to creator
            if creator_id not in self.revenue_by_creator:
                self.revenue_by_creator[creator_id] = 0.0
            self.revenue_by_creator[creator_id] += revenue

    def calculate_cpm_equivalent(self, creator_id: str,
                                 impressions: int) -> float:
        """
        Calculate effective CPM from labor revenue

        CPM = (Revenue / Impressions) * 1000
        """
        revenue = self.revenue_by_creator.get(creator_id, 0.0)
        if impressions == 0:
            return 0.0
        return (revenue / impressions) * 1000

    def compare_to_ads(self, creator_id: str,
                       ad_cpm: float, impressions: int) -> Dict[str, Any]:
        """
        Compare labor revenue to what ads would have generated
        """
        labor_revenue = self.revenue_by_creator.get(creator_id, 0.0)
        labor_cpm = self.calculate_cpm_equivalent(creator_id, impressions)

        ad_revenue_equivalent = (ad_cpm / 1000) * impressions

        return {
            "labor_revenue": labor_revenue,
            "labor_cpm": labor_cpm,
            "ad_revenue_equivalent": ad_revenue_equivalent,
            "ad_cpm": ad_cpm,
            "difference": labor_revenue - ad_revenue_equivalent,
            "difference_pct": ((labor_revenue - ad_revenue_equivalent) / ad_revenue_equivalent * 100)
                              if ad_revenue_equivalent > 0 else 0.0,
            "viable": labor_revenue >= ad_revenue_equivalent * 0.9  # Within 10%
        }
```

### 4.6 Analytics & Reporting

**File**: `src/analytics/metrics_collector.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, Any, List

class MetricsCollector:
    """
    Collects and reports simulation metrics
    """

    def __init__(self):
        self.sessions: list[ContentAccessSession] = []
        self.user_choices: list[str] = []
        self.quality_metrics = []

    def record_session(self, session: ContentAccessSession):
        """Record a content access session"""
        self.sessions.append(session)
        self.user_choices.append(session.choice)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""

        # Choice distribution
        choice_dist = Counter(self.user_choices)
        total = len(self.user_choices)

        # Convert sessions to DataFrame for analysis
        df = pd.DataFrame([s.dict() for s in self.sessions])

        report = {
            "summary": {
                "total_sessions": total,
                "choice_distribution": {
                    "labor_pct": (choice_dist.get("labor", 0) / total * 100) if total > 0 else 0,
                    "ads_pct": (choice_dist.get("ads", 0) / total * 100) if total > 0 else 0,
                    "payment_pct": (choice_dist.get("payment", 0) / total * 100) if total > 0 else 0
                }
            },
            "revenue": {
                "total": df["revenue_to_creator"].sum(),
                "by_source": df.groupby("revenue_source")["revenue_to_creator"].sum().to_dict(),
                "avg_per_session": df["revenue_to_creator"].mean()
            },
            "quality": {
                # Populate from quality control system
            },
            "timing": {
                "avg_task_time": df[df["choice"] == "labor"]["total_task_time_seconds"].mean(),
                "avg_decision_time": df["decision_time_ms"].mean()
            }
        }

        return report

    def visualize_results(self, output_dir: str = "./results"):
        """Generate visualization plots"""

        df = pd.DataFrame([s.dict() for s in self.sessions])

        # 1. Choice distribution pie chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        choice_counts = df["choice"].value_counts()
        axes[0, 0].pie(choice_counts.values, labels=choice_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title("User Choice Distribution")

        # 2. Revenue by source
        revenue_by_source = df.groupby("revenue_source")["revenue_to_creator"].sum()
        axes[0, 1].bar(revenue_by_source.index, revenue_by_source.values)
        axes[0, 1].set_title("Revenue by Source")
        axes[0, 1].set_ylabel("Revenue ($)")

        # 3. Task time distribution
        labor_sessions = df[df["choice"] == "labor"]
        axes[1, 0].hist(labor_sessions["total_task_time_seconds"], bins=30)
        axes[1, 0].set_title("Task Completion Time Distribution")
        axes[1, 0].set_xlabel("Time (seconds)")

        # 4. Quality metrics over time
        # Filter out None values for quality plotting
        quality_data = labor_sessions[labor_sessions["avg_label_quality"].notna()]
        if len(quality_data) > 0:
            axes[1, 1].plot(quality_data.index, quality_data["avg_label_quality"])
            axes[1, 1].set_title("Label Quality Over Time")
            axes[1, 1].set_ylabel("Fleiss' Kappa")
        else:
            axes[1, 1].text(0.5, 0.5, "No quality data available",
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/simulation_results.png", dpi=300)
        plt.close()
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

**Goal**: Set up foundation, load data, implement basic task interface

**Tasks**:
1. **Project Setup**
   - Initialize repository structure
   - Set up `uv` environment
   - Install dependencies
   - Configure logging

2. **Data Management**
   - Implement `DatasetManager`
   - Load MNIST, CIFAR-10, SST-2
   - Create `Task` generation from datasets
   - Test data loading and sampling

3. **Basic Schemas**
   - Implement all Pydantic models
   - Write unit tests for schemas
   - Validate serialization/deserialization

4. **Task Interface Simulation**
   - Create simple task presentation logic
   - Implement simulated user labeling (random + accuracy bias)
   - Test task completion flow

**Deliverables**:
- ✅ Working data loader for 3 datasets
- ✅ Task generation working
- ✅ Basic labeling simulation
- ✅ Unit tests passing

**Success Criteria**:
- Can load 1000 tasks from each dataset
- Can simulate 100 label submissions
- Code coverage >80%

---

### Phase 2: Agent Simulation & Marketplace (Weeks 3-4)

**Goal**: Implement agent-based model and marketplace dynamics

**Tasks**:
1. **Agent Implementation**
   - Implement `UserAgentMesa`
   - Implement `CreatorAgentMesa`
   - Implement `CompanyAgentMesa`
   - Create agent populations with configured distributions

2. **Mesa Model**
   - Create main simulation model
   - Implement step() functions
   - Set up scheduler (RandomActivation)
   - Configure data collection

3. **Marketplace Engine**
   - Implement task inventory management
   - Create task routing algorithm
   - Test task assignment

4. **Decision Logic**
   - Implement utility-based user choices
   - Calibrate decision weights
   - Test choice distribution matches expectations

**Deliverables**:
- ✅ Working agent-based model
- ✅ Marketplace routing tasks
- ✅ User agents making choices
- ✅ Simulation runs for N steps

**Success Criteria**:
- 1000 users, 100 creators, 10 companies
- Simulate 10,000 content access events
- Choice distribution within 10% of configured percentages
- No crashes or deadlocks

---

### Phase 3: Quality Control & Revenue (Weeks 5-6)

**Goal**: Implement quality validation and revenue calculation

**Tasks**:
1. **Quality Control System**
   - Implement consensus algorithms (majority vote)
   - Implement Fleiss' kappa calculation
   - Test accuracy validation
   - Implement quality-based payment

2. **Revenue Engine**
   - Implement revenue distribution
   - Calculate CPM equivalents
   - Track revenue by source
   - Implement comparison to ad revenue

3. **Advanced Quality Methods (Optional)**
   - Integrate crowd-kit library
   - Test Dawid-Skene algorithm
   - Implement LabelAId-style real-time QC

4. **Calibration**
   - Tune pricing to match CPM benchmarks
   - Adjust quality thresholds
   - Validate economic viability

**Deliverables**:
- ✅ Quality control working with 3+ redundancy
- ✅ Revenue calculated and distributed
- ✅ CPM comparisons showing viability range
- ✅ Quality metrics dashboard

**Success Criteria**:
- Fleiss' kappa >0.70 on average
- Accuracy >80% vs ground truth
- Labor CPM within 50% of ad CPM ($1.50-$12)
- Quality metrics match literature baselines

---

### Phase 4: Analysis & Documentation (Weeks 7-8)

**Goal**: Run experiments, analyze results, document findings

**Tasks**:
1. **Experimental Runs**
   - Run baseline simulation (10k events)
   - Run parameter sweeps:
     - Vary pricing ($0.01-$0.50 per label)
     - Vary user distributions (10-50% task preferers)
     - Vary quality requirements (0.60-0.80 kappa)
   - Collect results

2. **Analysis**
   - Statistical analysis of results
   - Compare to research questions
   - Identify insights and patterns
   - Generate publication-quality plots

3. **Metrics & Reporting**
   - Implement comprehensive metrics collector
   - Generate automated reports
   - Create visualizations
   - Export data for external analysis

4. **Documentation**
   - Write research paper/report
   - Document code (docstrings)
   - Create README with usage instructions
   - Record video demo (optional)

**Deliverables**:
- ✅ Experimental results from 5+ scenarios
- ✅ Statistical analysis report
- ✅ Visualizations and charts
- ✅ Research paper draft or technical report
- ✅ Complete code documentation

**Success Criteria**:
- Can answer all 5 core research questions (from CLAUDE.md)
- Results reproducible with fixed random seed
- Findings documented and visualized
- Code is clean, tested, documented

---

## 6. Configuration & Parameters

### 6.1 Configuration File Structure

**File**: `config/default.yaml`

```yaml
# Simulation Configuration

simulation:
  random_seed: 42
  num_steps: 10000  # Content access events to simulate
  output_dir: "./results"

datasets:
  mnist:
    enabled: true
    split: "test"
  cifar10:
    enabled: true
    split: "test"
  sst2:
    enabled: true
    split: "validation"

agents:
  users:
    count: 1000
    distribution:
      task_avoider: 0.40
      balanced: 0.35
      task_preferer: 0.15
      payment_preferer: 0.10

    behavioral_params:
      time_sensitivity:
        low: 0.33
        medium: 0.34
        high: 0.33
      base_accuracy: 0.85  # Average user accuracy on easy tasks

  creators:
    count: 100
    distribution:
      small: 0.50
      medium: 0.35
      large: 0.15

    size_params:
      small:
        monthly_visitors: [1000, 10000]
        current_cpm: [3.12, 5.00]
      medium:
        monthly_visitors: [10000, 100000]
        current_cpm: [4.00, 7.00]
      large:
        monthly_visitors: [100000, 1000000]
        current_cpm: [6.00, 8.60]

    labor_task_count: 5  # Tasks user must complete

  companies:
    count: 10
    distribution:
      startup: 0.40
      midsize: 0.40
      enterprise: 0.20

    pricing:
      easy: [0.01, 0.05]
      medium: [0.05, 0.15]
      hard: [0.15, 0.50]

    quality_requirements:
      startup: 0.65
      midsize: 0.70
      enterprise: 0.75

marketplace:
  redundancy_factor: 3  # Labels per task for consensus
  quality_bonus_multiplier: 1.5  # Bonus for >0.75 kappa

revenue:
  platform_share: 0.20
  creator_share: 0.70
  user_share: 0.10

quality_control:
  min_fleiss_kappa: 0.70
  consensus_method: "majority_vote"  # or "dawid_skene", "crowdlab"

logging:
  level: "INFO"
  file: "./logs/simulation.log"
```

### 6.2 Loading Configuration

**File**: `src/config/config_loader.py`

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has all required fields"""
    required_keys = ['simulation', 'agents', 'marketplace', 'revenue']
    return all(key in config for key in required_keys)
```

---

## 7. Quality Metrics

### 7.1 Label Quality Metrics

**Primary Metric**: Fleiss' Kappa
- **Target**: >0.75 (excellent agreement)
- **Good**: 0.60-0.75 (substantial agreement)
- **Acceptable**: 0.40-0.60 (moderate agreement)
- **Poor**: <0.40

**Secondary Metrics**:
- Accuracy vs. ground truth (target: >80%)
- Per-user accuracy distribution
- Per-task-type quality differences
- Quality degradation over time (fatigue)

### 7.2 Economic Viability Metrics

**Revenue Comparison**:
- Labor CPM vs. Ad CPM
- Total revenue: Labor vs. Ads vs. Payment
- Revenue per user
- Revenue per minute of user time

**Profitability**:
- Revenue after quality control filtering
- Platform margin
- Creator net revenue vs. current ads

### 7.3 User Behavior Metrics

**Choice Distribution**:
- % choosing Labor, Ads, Payment
- Choice by user type
- Choice by content value
- Choice by task difficulty

**Engagement**:
- Task completion rate
- Time per task
- Tasks per session
- User satisfaction proxy

### 7.4 System Performance Metrics

**Marketplace Efficiency**:
- Task assignment latency
- Inventory turnover rate
- Task completion rate
- Orphaned tasks (no takers)

**Quality Control Efficiency**:
- Labels required for consensus
- Rejected label rate
- Quality-adjusted labels per hour

---

## 8. Success Criteria

### 8.1 Technical Success Criteria

1. **System Functionality**
   - ✅ All datasets load without errors
   - ✅ Agents make decisions and complete tasks
   - ✅ Quality control computes accurate metrics
   - ✅ Revenue calculations are correct
   - ✅ Simulation runs to completion without crashes

2. **Performance**
   - ✅ Can simulate 10,000 events in <1 hour on CPU
   - ✅ Memory usage <4 GB
   - ✅ Results are deterministic with fixed seed

3. **Code Quality**
   - ✅ Unit test coverage >80%
   - ✅ All components have docstrings
   - ✅ Type hints throughout
   - ✅ Passes black formatting
   - ✅ No critical linting errors

### 8.2 Research Success Criteria

**Core Research Questions** (from CLAUDE.md):

1. **Will users choose data labeling over ads/payment?**
   - Success: >10% of users choose labor in baseline scenario
   - Can model different user distributions

2. **Can label revenue match advertising CPM?**
   - Success: Labor CPM within 50% of ad CPM ($1.50-$12)
   - Can identify parameter combinations that achieve parity

3. **Does access-motivated labeling produce quality comparable to paid crowdsourcing?**
   - Success: Fleiss' kappa >0.70 on average
   - Success: Accuracy >75% vs. ground truth
   - Comparable to literature baselines

4. **What task types and UX maximize acceptance and quality?**
   - Success: Can quantify quality by task type
   - Success: Can identify optimal task count (5-10 tasks)
   - Can model task difficulty tolerance

5. **What revenue split feels fair?**
   - Success: Can model different splits (platform/creator/user)
   - Can quantify impact on system sustainability

### 8.3 Deliverable Success Criteria

1. **Working Prototype**
   - ✅ Runs end-to-end simulation
   - ✅ Produces meaningful results
   - ✅ Generates reports and visualizations

2. **Documentation**
   - ✅ README with setup instructions
   - ✅ Code documentation (docstrings)
   - ✅ Research report or paper draft
   - ✅ Results visualization

3. **Reproducibility**
   - ✅ Random seed control works
   - ✅ Configuration files documented
   - ✅ Results can be replicated
   - ✅ All data sources cited

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Test Coverage**:
- Data schemas (Pydantic validation)
- Dataset loading and sampling
- Agent decision functions
- Quality control algorithms
- Revenue calculations
- Utility functions

**Framework**: pytest

**Example**:
```python
# tests/test_quality_control.py

def test_fleiss_kappa_perfect_agreement():
    labels = ["cat", "cat", "cat"]
    kappa = compute_fleiss_kappa(labels)
    assert kappa == 1.0

def test_fleiss_kappa_no_agreement():
    labels = ["cat", "dog", "bird"]
    kappa = compute_fleiss_kappa(labels)
    assert kappa < 0.5

def test_majority_vote():
    labels = ["cat", "cat", "dog"]
    consensus = majority_vote(labels)
    assert consensus == "cat"
```

### 9.2 Integration Tests

**Test Scenarios**:
- End-to-end simulation with small parameters
- Agent interactions with marketplace
- Quality control with multiple labels
- Revenue distribution flow

### 9.3 Validation Tests

**Validation Against Literature**:
- Quality metrics match expected ranges
- User choice distributions make sense
- Revenue calculations match examples
- CPM comparisons are reasonable

---

## 10. Reproducibility

### 10.1 Random Seed Management

```python
import random
import numpy as np
import torch

def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 10.2 Logging

**Log all key events**:
- Agent decisions
- Task assignments
- Label submissions
- Quality control results
- Revenue distributions

**Format**: Structured JSON logs for easy parsing

```python
import logging
import json

logger = logging.getLogger(__name__)

def log_event(event_type: str, data: Dict[str, Any]):
    logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }))
```

### 10.3 Results Archiving

**Save for each run**:
- Configuration file used
- Random seed
- Git commit hash
- Timestamp
- All output data
- Generated visualizations
- Summary report

**Directory structure**:
```
results/
  run_2025-11-23_14-30-00/
    config.yaml
    metadata.json
    sessions.csv
    labels.csv
    quality_metrics.json
    revenue_report.json
    plots/
      choice_distribution.png
      quality_over_time.png
      revenue_comparison.png
    report.pdf
```

---

## 11. File Structure

```
labor_content_access_model/
├── CLAUDE.md
├── RESEARCH.md
├── DATA_AVAILABILITY.md
├── IMPLEMENTATION.md (this file)
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── default.yaml
│   └── experiments/
│       ├── baseline.yaml
│       ├── high_labor_preference.yaml
│       └── low_pricing.yaml
│
├── src/
│   ├── __init__.py
│   ├── main.py (entry point)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_manager.py
│   │   └── schemas.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── user_agent.py
│   │   ├── creator_agent.py
│   │   └── company_agent.py
│   │
│   ├── marketplace/
│   │   ├── __init__.py
│   │   └── marketplace.py
│   │
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── quality_control.py
│   │   └── consensus.py
│   │
│   ├── revenue/
│   │   ├── __init__.py
│   │   └── revenue_engine.py
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics_collector.py
│   │   └── visualizer.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── model.py (Mesa model)
│   │
│   └── config/
│       ├── __init__.py
│       └── config_loader.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_agents.py
│   ├── test_quality_control.py
│   ├── test_revenue.py
│   └── test_simulation.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_calibration.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/
│   ├── run_baseline.sh
│   ├── run_experiments.sh
│   └── generate_report.py
│
├── results/
│   └── (generated during runs)
│
└── logs/
    └── (generated during runs)
```

---

## 12. Next Steps

1. **Review this implementation plan**
2. **Confirm approach and priorities**
3. **Begin Phase 1 implementation**
4. **Set up project structure and dependencies**
5. **Start coding!**

**Ready to begin implementation when you confirm with "proceed".**

---

**Questions for Clarification**:

1. Should we include LLM-based synthetic user decisions, or stick with rule-based utility functions for MVP?
2. Priority: Depth (fewer scenarios, more sophisticated) vs. Breadth (more scenarios, simpler implementation)?
3. Target audience for final report: Academic paper, blog post, or technical report?
4. Any specific research questions to prioritize?

**End of Implementation Plan**
