"""Tests for MarketplaceEngine"""

import pytest
from datetime import datetime
from src.marketplace.marketplace import MarketplaceEngine
from src.data.schemas import Task, TaskType, TaskDifficulty


@pytest.fixture
def config():
    """Test configuration"""
    return {
        "marketplace": {
            "redundancy_factor": 3
        }
    }


@pytest.fixture
def marketplace(config):
    """Create MarketplaceEngine instance"""
    return MarketplaceEngine(config)


@pytest.fixture
def sample_tasks():
    """Create sample tasks with different difficulties"""
    tasks = []
    for i, difficulty in enumerate([TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]):
        task = Task(
            task_id=f"task-{i}",
            task_type=TaskType.IMAGE_CLASSIFICATION,
            difficulty=difficulty,
            dataset_name="mnist",
            item_index=i,
            item_data={"type": "image"},
            ground_truth_label=str(i),
            possible_labels=["0", "1", "2", "3", "4"],
            company_id="company-1",
            created_at=datetime.now(),
            estimated_time_seconds=10.0,
            price_per_label=0.10,
            quality_requirement=0.70
        )
        tasks.append(task)
    return tasks


def test_marketplace_initialization(marketplace):
    """Test MarketplaceEngine initializes correctly"""
    assert marketplace.task_inventory == {}
    assert marketplace.task_queue == []
    assert marketplace.total_tasks_created == 0


def test_add_task(marketplace, sample_tasks):
    """Test adding tasks to marketplace"""
    task = sample_tasks[0]
    marketplace.add_task(task)

    assert len(marketplace.task_inventory) == 1
    assert len(marketplace.task_queue) == 1
    assert marketplace.total_tasks_created == 1
    assert task.task_id in marketplace.task_inventory


def test_assign_tasks(marketplace, sample_tasks):
    """Test assigning tasks to users"""
    # Add tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign 2 tasks
    assigned = marketplace.assign_tasks(user_id="user-1", count=2)

    assert len(assigned) == 2
    assert marketplace.total_tasks_assigned == 2
    assert len(marketplace.task_queue) == 1  # 1 task left


def test_assign_tasks_insufficient_inventory(marketplace, sample_tasks):
    """Test assigning more tasks than available"""
    # Add only 2 tasks
    for task in sample_tasks[:2]:
        marketplace.add_task(task)

    # Try to assign 5 tasks
    assigned = marketplace.assign_tasks(user_id="user-1", count=5)

    # Should only get 2
    assert len(assigned) == 2


def test_filter_by_difficulty_easy_only(marketplace, sample_tasks):
    """Test filtering tasks by difficulty (easy only)"""
    # Add all tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign with easy_only filter
    assigned = marketplace.assign_tasks(
        user_id="user-1",
        count=10,
        difficulty_tolerance="easy_only"
    )

    # Should only get the easy task
    assert len(assigned) == 1
    assert assigned[0].difficulty == TaskDifficulty.EASY


def test_filter_by_difficulty_medium(marketplace, sample_tasks):
    """Test filtering tasks by difficulty (medium tolerance)"""
    # Add all tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign with medium filter (includes easy and medium)
    assigned = marketplace.assign_tasks(
        user_id="user-1",
        count=10,
        difficulty_tolerance="medium"
    )

    # Should get easy and medium, not hard
    assert len(assigned) == 2
    difficulties = [t.difficulty for t in assigned]
    assert TaskDifficulty.EASY in difficulties
    assert TaskDifficulty.MEDIUM in difficulties
    assert TaskDifficulty.HARD not in difficulties


def test_filter_by_difficulty_all(marketplace, sample_tasks):
    """Test filtering tasks by difficulty (all)"""
    # Add all tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign with all filter
    assigned = marketplace.assign_tasks(
        user_id="user-1",
        count=10,
        difficulty_tolerance="all"
    )

    # Should get all tasks
    assert len(assigned) == 3


def test_mark_task_completed(marketplace, sample_tasks):
    """Test marking task as completed"""
    task = sample_tasks[0]
    marketplace.add_task(task)

    assert len(marketplace.task_queue) == 1

    marketplace.mark_task_completed(task.task_id)

    assert len(marketplace.task_queue) == 0
    assert marketplace.total_tasks_completed == 1


def test_get_task(marketplace, sample_tasks):
    """Test getting task by ID"""
    task = sample_tasks[0]
    marketplace.add_task(task)

    retrieved = marketplace.get_task(task.task_id)

    assert retrieved is not None
    assert retrieved.task_id == task.task_id


def test_get_task_statistics(marketplace, sample_tasks):
    """Test getting marketplace statistics"""
    # Add tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign one
    marketplace.assign_tasks(user_id="user-1", count=1)

    stats = marketplace.get_task_statistics()

    assert stats['total_tasks_created'] == 3
    assert stats['total_tasks_assigned'] == 1
    assert stats['available_tasks'] == 2
    assert 'tasks_by_difficulty' in stats
    assert 'avg_price_per_label' in stats


def test_clear_completed_tasks(marketplace, sample_tasks):
    """Test clearing completed tasks from inventory"""
    # Add tasks
    for task in sample_tasks:
        marketplace.add_task(task)

    # Assign all tasks (removes from queue)
    marketplace.assign_tasks(user_id="user-1", count=10)

    assert len(marketplace.task_inventory) == 3
    assert len(marketplace.task_queue) == 0

    # Clear completed
    marketplace.clear_completed_tasks()

    # Should remove tasks not in queue
    assert len(marketplace.task_inventory) == 0
