"""Company agent implementation for agent-based modeling"""

import random
from mesa import Agent
from src.data.schemas import CompanyAgent, CompanySize, TaskType, TaskDifficulty


class CompanyAgentMesa(Agent):
    """
    Mesa-based company agent

    Represents an ML company needing labeled data
    """

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
        # Assuming 30 days per month, calculate daily need
        labels_per_step = self.company_data.labels_needed_per_month / 30

        # Account for redundancy (need N labels per item)
        tasks_to_create = int(labels_per_step / self.company_data.redundancy_factor)

        # Create tasks
        for _ in range(max(1, tasks_to_create)):  # At least 1 task per step
            task = self.create_task()
            if task:
                self.model.marketplace.add_task(task)
                self.company_data.available_tasks.append(task.task_id)

    def create_task(self):
        """
        Create a labeling task from this company's needs

        Returns:
            Task object or None if no dataset manager available
        """
        if not hasattr(self.model, 'dataset_manager'):
            return None

        # Sample from task types needed
        task_type = random.choice(self.company_data.task_types_needed)

        # Map task type to dataset
        dataset_name = self.get_dataset_for_type(task_type)

        if dataset_name not in self.model.dataset_manager.datasets:
            return None

        # Generate task from dataset
        task = self.model.dataset_manager.sample_task(
            dataset_name=dataset_name,
            task_type=task_type,
            company_id=self.company_data.company_id
        )

        # Set pricing based on difficulty
        task.price_per_label = self.get_price_for_difficulty(task.difficulty)

        # Set quality requirement
        task.quality_requirement = self.company_data.min_quality_kappa

        return task

    def get_dataset_for_type(self, task_type: TaskType) -> str:
        """
        Map task type to dataset name

        Args:
            task_type: TaskType enum

        Returns:
            Dataset name string
        """
        mapping = {
            TaskType.IMAGE_CLASSIFICATION: random.choice(['mnist', 'cifar10']),
            TaskType.TEXT_SENTIMENT: 'sst2',
            TaskType.NAMED_ENTITY_RECOGNITION: 'sst2',  # Fallback to sst2
            TaskType.AUDIO_EMOTION: 'sst2'  # Fallback to sst2
        }
        return mapping.get(task_type, 'mnist')

    def get_price_for_difficulty(self, difficulty: TaskDifficulty) -> float:
        """
        Get price per label for given difficulty

        Args:
            difficulty: TaskDifficulty enum

        Returns:
            Price per label
        """
        price_range = self.company_data.max_price_per_label.get(
            difficulty,
            0.10  # Default price
        )

        # If price_range is a dict entry, return it directly
        if isinstance(price_range, (int, float)):
            return float(price_range)

        # Otherwise shouldn't happen, but return default
        return 0.10

    def get_stats(self):
        """
        Get company statistics

        Returns:
            Dictionary of company stats
        """
        return {
            "company_id": self.company_data.company_id,
            "size": self.company_data.size.value,
            "total_labels_purchased": self.company_data.total_labels_purchased,
            "total_spend": self.company_data.total_spend,
            "avg_quality_received": self.company_data.avg_quality_received,
            "available_tasks": len(self.company_data.available_tasks),
            "completed_tasks": len(self.company_data.completed_tasks)
        }
