"""Tests for DatasetManager"""

import pytest
from src.data.dataset_manager import DatasetManager
from src.data.schemas import TaskType, TaskDifficulty


@pytest.fixture
def config():
    """Test configuration"""
    return {
        "datasets": {
            "mnist": {"enabled": True, "split": "test"},
            "cifar10": {"enabled": True, "split": "test"},
            "sst2": {"enabled": True, "split": "validation"}
        }
    }


@pytest.fixture
def dataset_manager(config):
    """Create DatasetManager instance"""
    manager = DatasetManager(config)
    return manager


def test_dataset_manager_initialization(dataset_manager):
    """Test DatasetManager initializes correctly"""
    assert dataset_manager.config is not None
    assert dataset_manager.datasets == {}
    assert dataset_manager.ground_truth == {}


def test_difficulty_mapping(dataset_manager):
    """Test _get_difficulty method"""
    assert dataset_manager._get_difficulty('mnist') == TaskDifficulty.EASY
    assert dataset_manager._get_difficulty('cifar10') == TaskDifficulty.MEDIUM
    assert dataset_manager._get_difficulty('sst2') == TaskDifficulty.MEDIUM
    assert dataset_manager._get_difficulty('unknown') == TaskDifficulty.MEDIUM  # Default


def test_time_estimation(dataset_manager):
    """Test _estimate_task_time method"""
    # MNIST easy task
    time_easy = dataset_manager._estimate_task_time('mnist', TaskDifficulty.EASY)
    assert 5.0 < time_easy < 15.0

    # CIFAR-10 medium task
    time_medium = dataset_manager._estimate_task_time('cifar10', TaskDifficulty.MEDIUM)
    assert 10.0 < time_medium < 25.0

    # Hard task should take longer
    time_hard = dataset_manager._estimate_task_time('cifar10', TaskDifficulty.HARD)
    assert time_hard > time_medium


# Note: The following tests require downloading datasets, so they are marked as slow
@pytest.mark.slow
def test_load_datasets(dataset_manager):
    """Test loading datasets (requires internet)"""
    dataset_manager.load_datasets()

    assert 'mnist' in dataset_manager.datasets
    assert 'cifar10' in dataset_manager.datasets
    assert 'sst2' in dataset_manager.datasets

    # Check label mappings
    assert len(dataset_manager.label_mappings['mnist']) == 10
    assert len(dataset_manager.label_mappings['cifar10']) == 10
    assert len(dataset_manager.label_mappings['sst2']) == 2


@pytest.mark.slow
def test_sample_task(dataset_manager):
    """Test sampling a task from dataset"""
    dataset_manager.load_datasets()

    task = dataset_manager.sample_task(
        dataset_name='mnist',
        task_type=TaskType.IMAGE_CLASSIFICATION,
        company_id='test-company',
        price_per_label=0.05
    )

    assert task.dataset_name == 'mnist'
    assert task.task_type == TaskType.IMAGE_CLASSIFICATION
    assert task.difficulty == TaskDifficulty.EASY
    assert task.company_id == 'test-company'
    assert task.price_per_label == 0.05

    # Check that ground_truth_label is a string
    assert isinstance(task.ground_truth_label, str)
    # Check that possible_labels contains strings
    assert all(isinstance(label, str) for label in task.possible_labels)
    # Check that ground_truth is in possible_labels
    assert task.ground_truth_label in task.possible_labels


@pytest.mark.slow
def test_get_task_at_index(dataset_manager):
    """Test getting specific task by index"""
    dataset_manager.load_datasets()

    task = dataset_manager.get_task_at_index(
        dataset_name='mnist',
        index=0,
        task_type=TaskType.IMAGE_CLASSIFICATION,
        company_id='test-company'
    )

    assert task.item_index == 0
    assert task.dataset_name == 'mnist'
    assert isinstance(task.ground_truth_label, str)


@pytest.mark.slow
def test_get_dataset_info(dataset_manager):
    """Test getting dataset information"""
    dataset_manager.load_datasets()

    info = dataset_manager.get_dataset_info()

    assert 'mnist' in info
    assert 'cifar10' in info
    assert 'sst2' in info

    # Check MNIST info
    mnist_info = info['mnist']
    assert 'splits' in mnist_info
    assert 'num_examples' in mnist_info
    assert 'num_labels' in mnist_info
    assert mnist_info['num_labels'] == 10


@pytest.mark.slow
def test_type_consistency(dataset_manager):
    """Test that ground_truth_label and possible_labels have consistent types"""
    dataset_manager.load_datasets()

    for dataset_name in ['mnist', 'cifar10', 'sst2']:
        task = dataset_manager.sample_task(
            dataset_name=dataset_name,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            company_id='test-company'
        )

        # Both should be strings
        assert isinstance(task.ground_truth_label, str)
        assert all(isinstance(label, str) for label in task.possible_labels)

        # ground_truth should be in possible_labels
        assert task.ground_truth_label in task.possible_labels
