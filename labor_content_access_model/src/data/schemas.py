"""Data schemas for the labor-for-content-access system"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """Types of labeling tasks"""
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_SENTIMENT = "text_sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    AUDIO_EMOTION = "audio_emotion"


class TaskDifficulty(str, Enum):
    """Difficulty levels for tasks"""
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
    possible_labels: List[str]  # e.g., ["0", "1", ..., "9"] for MNIST

    # Metadata
    company_id: str  # Which company needs this labeled
    created_at: datetime
    estimated_time_seconds: float  # Expected completion time

    # Pricing
    price_per_label: float  # What company pays per label
    quality_requirement: float  # Minimum Fleiss' kappa required

    class Config:
        arbitrary_types_allowed = True


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

    class Config:
        arbitrary_types_allowed = True


class UserType(str, Enum):
    """Types of user personas"""
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
    privacy_concern: Literal["low", "medium", "high"] = "medium"

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

    # Note: difficulty_penalty should be set in __init__ to avoid mutable default issues
    difficulty_penalty: Optional[Dict[TaskDifficulty, float]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.difficulty_penalty is None:
            self.difficulty_penalty = {
                TaskDifficulty.EASY: 0.0,
                TaskDifficulty.MEDIUM: 0.1,
                TaskDifficulty.HARD: 0.2
            }


class CreatorSize(str, Enum):
    """Creator size categories"""
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


class CompanySize(str, Enum):
    """Company size categories"""
    STARTUP = "startup"
    MIDSIZE = "midsize"
    ENTERPRISE = "enterprise"


class CompanyAgent(BaseModel):
    """Represents an ML company needing labeled data"""

    company_id: str
    size: CompanySize

    # Data needs
    task_types_needed: List[TaskType]
    monthly_label_budget: float  # How much they'll pay per month
    labels_needed_per_month: int

    # Quality requirements
    min_quality_kappa: float = 0.70  # Minimum Fleiss' kappa
    redundancy_factor: int = 3  # How many labels per item (for consensus)

    # Pricing
    max_price_per_label: Dict[TaskDifficulty, float] = Field(
        default_factory=lambda: {
            TaskDifficulty.EASY: 0.05,
            TaskDifficulty.MEDIUM: 0.15,
            TaskDifficulty.HARD: 0.50
        }
    )
    quality_bonus_multiplier: float = 1.5  # Pay 1.5x for >0.75 kappa

    # Inventory
    available_tasks: List[str] = Field(default_factory=list)  # List of task_ids
    completed_tasks: List[str] = Field(default_factory=list)

    # Tracking
    total_labels_purchased: int = 0
    total_spend: float = 0.0
    avg_quality_received: float = 0.0


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
    tasks_assigned: Optional[List[str]] = None
    tasks_completed: Optional[List[str]] = None
    labels_submitted: Optional[List[str]] = None
    total_task_time_seconds: Optional[float] = None
    avg_label_quality: Optional[float] = None

    # Revenue
    revenue_to_creator: float = 0.0
    revenue_to_platform: float = 0.0
    revenue_source: Optional[str] = None  # "labor", "ads", "payment"

    # Outcome
    access_granted: bool
    user_satisfaction_proxy: Optional[float] = None  # Simulated satisfaction score
