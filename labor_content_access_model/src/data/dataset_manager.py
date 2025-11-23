"""Dataset management for loading and sampling labeling tasks"""

import random
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from datasets import load_dataset
from .schemas import Task, TaskType, TaskDifficulty


class DatasetManager:
    """
    Manages loading and sampling from datasets (MNIST, CIFAR-10, SST-2)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets = {}
        self.ground_truth = {}
        self.label_mappings = {}

    def load_datasets(self):
        """Load all configured datasets from Hugging Face"""
        print("Loading datasets from Hugging Face...")

        # Load MNIST
        if self.config.get("datasets", {}).get("mnist", {}).get("enabled", True):
            print("  Loading MNIST...")
            self.datasets['mnist'] = load_dataset("mnist")
            self.label_mappings['mnist'] = [str(i) for i in range(10)]
            self.ground_truth['mnist'] = {
                'train': self.datasets['mnist']['train']['label'],
                'test': self.datasets['mnist']['test']['label']
            }

        # Load CIFAR-10
        if self.config.get("datasets", {}).get("cifar10", {}).get("enabled", True):
            print("  Loading CIFAR-10...")
            self.datasets['cifar10'] = load_dataset("cifar10")
            self.label_mappings['cifar10'] = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            self.ground_truth['cifar10'] = {
                'train': self.datasets['cifar10']['train']['label'],
                'test': self.datasets['cifar10']['test']['label']
            }

        # Load SST-2
        if self.config.get("datasets", {}).get("sst2", {}).get("enabled", True):
            print("  Loading SST-2...")
            self.datasets['sst2'] = load_dataset("stanfordnlp/sst2")
            self.label_mappings['sst2'] = ['negative', 'positive']
            self.ground_truth['sst2'] = {
                'train': self.datasets['sst2']['train']['label'],
                'validation': self.datasets['sst2']['validation']['label']
            }

        print(f"Loaded {len(self.datasets)} datasets successfully!")

    def sample_task(
        self,
        dataset_name: str,
        task_type: TaskType,
        company_id: str,
        split: str = "test",
        price_per_label: float = 0.10,
        quality_requirement: float = 0.70
    ) -> Task:
        """
        Sample a random item from dataset and create a Task

        Args:
            dataset_name: Name of dataset ('mnist', 'cifar10', 'sst2')
            task_type: Type of task
            company_id: ID of company requesting the task
            split: Dataset split to sample from
            price_per_label: Payment per label
            quality_requirement: Minimum quality (Fleiss' kappa) required

        Returns:
            Task object
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        # Adjust split name for SST-2 (uses 'validation' instead of 'test')
        if dataset_name == 'sst2' and split == 'test':
            split = 'validation'

        dataset_split = self.datasets[dataset_name][split]

        # Sample random index
        index = random.randint(0, len(dataset_split) - 1)

        return self.get_task_at_index(
            dataset_name=dataset_name,
            index=index,
            task_type=task_type,
            company_id=company_id,
            split=split,
            price_per_label=price_per_label,
            quality_requirement=quality_requirement
        )

    def get_task_at_index(
        self,
        dataset_name: str,
        index: int,
        task_type: TaskType,
        company_id: str,
        split: str = "test",
        price_per_label: float = 0.10,
        quality_requirement: float = 0.70
    ) -> Task:
        """
        Get specific item from dataset and create a Task

        Args:
            dataset_name: Name of dataset
            index: Index in the dataset
            task_type: Type of task
            company_id: ID of company requesting the task
            split: Dataset split
            price_per_label: Payment per label
            quality_requirement: Minimum quality required

        Returns:
            Task object
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        # Adjust split name for SST-2
        if dataset_name == 'sst2' and split == 'test':
            split = 'validation'

        dataset_split = self.datasets[dataset_name][split]
        item = dataset_split[index]

        # Determine difficulty
        difficulty = self._get_difficulty(dataset_name)

        # Get ground truth label (convert to string for consistency)
        ground_truth_label_int = item['label']
        ground_truth_label = str(ground_truth_label_int)

        # Get possible labels
        possible_labels = self.label_mappings[dataset_name]

        # Prepare item data (store the data needed for presentation)
        if dataset_name == 'mnist':
            item_data = {
                'type': 'image',
                'image': item['image'],  # PIL Image
                'dataset': 'mnist',
                'description': 'Handwritten digit classification (0-9)'
            }
        elif dataset_name == 'cifar10':
            item_data = {
                'type': 'image',
                'image': item['image'],  # PIL Image
                'dataset': 'cifar10',
                'description': 'Object classification in color images'
            }
        elif dataset_name == 'sst2':
            item_data = {
                'type': 'text',
                'text': item['sentence'],
                'dataset': 'sst2',
                'description': 'Sentiment analysis (positive/negative)'
            }
        else:
            item_data = {'raw': str(item)}

        # Estimate time (in seconds)
        estimated_time = self._estimate_task_time(dataset_name, difficulty)

        # Create task
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            difficulty=difficulty,
            dataset_name=dataset_name,
            item_index=index,
            item_data=item_data,
            ground_truth_label=ground_truth_label,
            possible_labels=possible_labels,
            company_id=company_id,
            created_at=datetime.now(),
            estimated_time_seconds=estimated_time,
            price_per_label=price_per_label,
            quality_requirement=quality_requirement
        )

        return task

    def _get_difficulty(self, dataset_name: str) -> TaskDifficulty:
        """Determine task difficulty based on dataset"""
        difficulty_map = {
            'mnist': TaskDifficulty.EASY,
            'cifar10': TaskDifficulty.MEDIUM,
            'sst2': TaskDifficulty.MEDIUM,
        }
        return difficulty_map.get(dataset_name, TaskDifficulty.MEDIUM)

    def _estimate_task_time(self, dataset_name: str, difficulty: TaskDifficulty) -> float:
        """
        Estimate time to complete task in seconds

        Based on typical crowdsourcing benchmarks:
        - MNIST: ~5-10 seconds
        - CIFAR-10: ~10-15 seconds
        - SST-2: ~10-20 seconds
        """
        base_times = {
            'mnist': 7.5,
            'cifar10': 12.5,
            'sst2': 15.0,
        }

        difficulty_multipliers = {
            TaskDifficulty.EASY: 1.0,
            TaskDifficulty.MEDIUM: 1.2,
            TaskDifficulty.HARD: 1.5,
        }

        base_time = base_times.get(dataset_name, 10.0)
        multiplier = difficulty_multipliers.get(difficulty, 1.0)

        return base_time * multiplier

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about loaded datasets"""
        info = {}
        for name, ds in self.datasets.items():
            info[name] = {
                'splits': list(ds.keys()),
                'num_examples': {split: len(ds[split]) for split in ds.keys()},
                'num_labels': len(self.label_mappings[name]),
                'labels': self.label_mappings[name]
            }
        return info
