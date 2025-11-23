"""Tests for QualityControlSystem"""

import pytest
from datetime import datetime
import uuid
from src.quality.quality_control import QualityControlSystem
from src.data.schemas import Label, Task, TaskType, TaskDifficulty


@pytest.fixture
def config():
    """Test configuration"""
    return {
        "marketplace": {
            "redundancy_factor": 3,
            "quality_bonus_multiplier": 1.5
        }
    }


@pytest.fixture
def quality_control(config):
    """Create QualityControlSystem instance"""
    return QualityControlSystem(config)


@pytest.fixture
def sample_task():
    """Create a sample task"""
    return Task(
        task_id="task-1",
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
        price_per_label=0.10,
        quality_requirement=0.70
    )


def test_quality_control_initialization(quality_control):
    """Test QualityControlSystem initializes correctly"""
    assert quality_control.redundancy_factor == 3
    assert quality_control.task_labels == {}
    assert quality_control.total_labels_processed == 0


def test_compute_fleiss_kappa_perfect_agreement(quality_control):
    """Test Fleiss' kappa with perfect agreement"""
    labels = ["5", "5", "5"]
    kappa = quality_control.compute_fleiss_kappa(labels)
    assert kappa == 1.0


def test_compute_fleiss_kappa_no_agreement(quality_control):
    """Test Fleiss' kappa with no agreement"""
    labels = ["0", "1", "2"]
    kappa = quality_control.compute_fleiss_kappa(labels)
    assert kappa < 0.5


def test_compute_fleiss_kappa_partial_agreement(quality_control):
    """Test Fleiss' kappa with partial agreement"""
    labels = ["5", "5", "3"]  # 2/3 agreement
    kappa = quality_control.compute_fleiss_kappa(labels)
    assert 0.0 < kappa < 1.0


def test_process_labels_insufficient_redundancy(quality_control, sample_task):
    """Test processing labels with insufficient redundancy"""
    quality_control.tasks["task-1"] = sample_task

    labels = [
        Label(
            label_id=str(uuid.uuid4()),
            task_id="task-1",
            user_id=f"user-{i}",
            submitted_label="5",
            time_taken_seconds=8.0,
            submitted_at=datetime.now()
        )
        for i in range(2)  # Only 2 labels, need 3
    ]

    revenue = quality_control.process_labels(labels)

    # Should not pay yet (need 3 labels)
    assert revenue == 0.0
    assert quality_control.total_labels_processed == 2
    assert "task-1" in quality_control.task_labels
    assert len(quality_control.task_labels["task-1"]) == 2


def test_process_labels_meets_quality(quality_control, sample_task):
    """Test processing labels that meet quality requirement"""
    quality_control.tasks["task-1"] = sample_task

    # 3 labels with good agreement (correct answer)
    labels = [
        Label(
            label_id=str(uuid.uuid4()),
            task_id="task-1",
            user_id=f"user-{i}",
            submitted_label="5",  # Correct
            time_taken_seconds=8.0,
            submitted_at=datetime.now()
        )
        for i in range(3)
    ]

    revenue = quality_control.process_labels(labels)

    # Should pay because kappa = 1.0 > 0.70
    expected_base = 0.10 * 3  # 3 labels at $0.10 each
    expected_with_bonus = expected_base * 1.5  # Bonus for kappa > 0.75
    assert revenue == pytest.approx(expected_with_bonus, rel=1e-6)


def test_process_labels_fails_quality(quality_control, sample_task):
    """Test processing labels that fail quality requirement"""
    # Set high quality requirement
    sample_task.quality_requirement = 0.90
    quality_control.tasks["task-1"] = sample_task

    # 3 labels with poor agreement
    labels = [
        Label(
            label_id=str(uuid.uuid4()),
            task_id="task-1",
            user_id=f"user-{i}",
            submitted_label=str(i),  # All different
            time_taken_seconds=8.0,
            submitted_at=datetime.now()
        )
        for i in range(3)
    ]

    revenue = quality_control.process_labels(labels)

    # Should not pay due to low quality
    assert revenue == 0.0


def test_get_quality_metrics(quality_control, sample_task):
    """Test getting quality metrics"""
    quality_control.tasks["task-1"] = sample_task

    # Process some labels
    labels = [
        Label(
            label_id=str(uuid.uuid4()),
            task_id="task-1",
            user_id=f"user-{i}",
            submitted_label="5",
            time_taken_seconds=8.0,
            submitted_at=datetime.now()
        )
        for i in range(3)
    ]

    quality_control.process_labels(labels)

    metrics = quality_control.get_quality_metrics()

    assert metrics['avg_fleiss_kappa'] == 1.0  # Perfect agreement
    assert metrics['avg_accuracy'] == 1.0  # Correct answer
    assert metrics['tasks_with_consensus'] == 1
    assert metrics['total_labels_processed'] == 3
    assert metrics['kappa_distribution']['excellent (>0.75)'] == 1


def test_label_correctness_tracking(quality_control, sample_task):
    """Test that label correctness is tracked"""
    quality_control.tasks["task-1"] = sample_task

    labels = [
        Label(
            label_id=str(uuid.uuid4()),
            task_id="task-1",
            user_id=f"user-{i}",
            submitted_label="5" if i < 2 else "3",  # 2 correct, 1 wrong
            time_taken_seconds=8.0,
            submitted_at=datetime.now()
        )
        for i in range(3)
    ]

    quality_control.process_labels(labels)

    # Check that is_correct was set
    assert labels[0].is_correct is True
    assert labels[1].is_correct is True
    assert labels[2].is_correct is False
