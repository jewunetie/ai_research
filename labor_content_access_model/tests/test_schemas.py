"""Tests for Pydantic schemas"""

import pytest
from datetime import datetime
from src.data.schemas import (
    Task, Label, UserAgent, CreatorAgent, CompanyAgent,
    TaskType, TaskDifficulty, UserType, CreatorSize, CompanySize,
    ContentAccessSession
)


def test_task_creation():
    """Test creating a Task object"""
    task = Task(
        task_id="test-123",
        task_type=TaskType.IMAGE_CLASSIFICATION,
        difficulty=TaskDifficulty.EASY,
        dataset_name="mnist",
        item_index=0,
        item_data={"type": "image"},
        ground_truth_label="5",
        possible_labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        company_id="company-1",
        created_at=datetime.now(),
        estimated_time_seconds=10.0,
        price_per_label=0.05,
        quality_requirement=0.70
    )

    assert task.task_id == "test-123"
    assert task.difficulty == TaskDifficulty.EASY
    assert task.ground_truth_label == "5"
    assert len(task.possible_labels) == 10


def test_label_creation():
    """Test creating a Label object"""
    label = Label(
        label_id="label-1",
        task_id="task-1",
        user_id="user-1",
        submitted_label="5",
        time_taken_seconds=8.5,
        submitted_at=datetime.now()
    )

    assert label.submitted_label == "5"
    assert label.is_correct is None  # Not yet validated
    assert label.quality_score is None


def test_user_agent_creation():
    """Test creating a UserAgent with default difficulty_penalty"""
    user = UserAgent(
        user_id="user-1",
        user_type=UserType.BALANCED,
        time_sensitivity="medium",
        task_difficulty_tolerance="all",
        ad_tolerance=0.5
    )

    assert user.user_id == "user-1"
    assert user.user_type == UserType.BALANCED
    # Check that difficulty_penalty was set by model_validator
    assert user.difficulty_penalty is not None
    assert TaskDifficulty.EASY in user.difficulty_penalty
    assert user.difficulty_penalty[TaskDifficulty.EASY] == 0.0
    assert user.difficulty_penalty[TaskDifficulty.MEDIUM] == 0.1


def test_user_agent_custom_difficulty_penalty():
    """Test creating UserAgent with custom difficulty_penalty"""
    custom_penalty = {
        TaskDifficulty.EASY: 0.05,
        TaskDifficulty.MEDIUM: 0.15,
        TaskDifficulty.HARD: 0.30
    }

    user = UserAgent(
        user_id="user-2",
        user_type=UserType.TASK_PREFERER,
        time_sensitivity="low",
        task_difficulty_tolerance="all",
        ad_tolerance=0.2,
        difficulty_penalty=custom_penalty
    )

    assert user.difficulty_penalty == custom_penalty
    assert user.difficulty_penalty[TaskDifficulty.EASY] == 0.05


def test_creator_agent_creation():
    """Test creating a CreatorAgent"""
    creator = CreatorAgent(
        creator_id="creator-1",
        size=CreatorSize.MEDIUM,
        monthly_visitors=50000,
        avg_visitors_per_day=1667,
        current_cpm=5.5,
        current_monthly_ad_revenue=275.0,
        content_value=7.5
    )

    assert creator.size == CreatorSize.MEDIUM
    assert creator.monthly_visitors == 50000
    assert creator.current_cpm == 5.5
    assert creator.total_access_events == 0  # Initial value


def test_company_agent_creation():
    """Test creating a CompanyAgent"""
    company = CompanyAgent(
        company_id="company-1",
        size=CompanySize.STARTUP,
        task_types_needed=[TaskType.IMAGE_CLASSIFICATION],
        monthly_label_budget=1000.0,
        labels_needed_per_month=10000
    )

    assert company.size == CompanySize.STARTUP
    assert len(company.task_types_needed) == 1
    assert company.min_quality_kappa == 0.70  # Default
    assert company.redundancy_factor == 3  # Default
    # Check that max_price_per_label was set by default_factory
    assert TaskDifficulty.EASY in company.max_price_per_label


def test_content_access_session():
    """Test creating a ContentAccessSession"""
    session = ContentAccessSession(
        session_id="session-1",
        timestamp=datetime.now(),
        user_id="user-1",
        creator_id="creator-1",
        choice="labor",
        decision_time_ms=1500.0,
        tasks_assigned=["task-1", "task-2"],
        labels_submitted=["label-1", "label-2"],
        total_task_time_seconds=45.5,
        revenue_to_creator=0.35,
        revenue_to_platform=0.10,
        revenue_source="labor",
        access_granted=True
    )

    assert session.choice == "labor"
    assert len(session.tasks_assigned) == 2
    assert session.access_granted is True


def test_enum_values():
    """Test that enums have correct values"""
    assert TaskType.IMAGE_CLASSIFICATION.value == "image_classification"
    assert TaskDifficulty.EASY.value == "easy"
    assert UserType.TASK_PREFERER.value == "task_preferer"
    assert CreatorSize.MEDIUM.value == "medium"
    assert CompanySize.ENTERPRISE.value == "enterprise"
