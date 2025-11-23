"""Marketplace engine for managing task inventory and routing"""

from typing import Dict, Any, List, Optional
from src.data.schemas import Task, TaskDifficulty, TaskType


class MarketplaceEngine:
    """
    Manages task inventory and routing between companies and users
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_inventory: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # Queue of available task_ids

        # Statistics
        self.total_tasks_created = 0
        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0

    def add_task(self, task: Task):
        """
        Company adds task to marketplace

        Args:
            task: Task to add
        """
        self.task_inventory[task.task_id] = task
        self.task_queue.append(task.task_id)
        self.total_tasks_created += 1

    def assign_tasks(
        self,
        user_id: str,
        count: int,
        difficulty_tolerance: Optional[str] = None
    ) -> List[Task]:
        """
        Assign N tasks to a user

        Routing strategy:
        - Consider user's difficulty tolerance (if provided)
        - Balance load across companies
        - Prioritize high-value tasks
        - Simple FIFO for MVP

        Args:
            user_id: ID of user requesting tasks
            count: Number of tasks to assign
            difficulty_tolerance: User's difficulty tolerance
                                 ("easy_only", "medium", "all")

        Returns:
            List of assigned tasks
        """
        assigned_tasks = []

        # Filter available tasks based on difficulty tolerance
        available_task_ids = self._filter_by_difficulty(difficulty_tolerance)

        for _ in range(count):
            if not available_task_ids:
                break

            # Simple FIFO for MVP
            # Could implement sophisticated routing later:
            # - Match task difficulty to user skill
            # - Balance across companies
            # - Prioritize high-paying tasks
            task_id = available_task_ids.pop(0)
            task = self.task_inventory[task_id]
            assigned_tasks.append(task)

            # Remove from main queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)

            self.total_tasks_assigned += 1

        return assigned_tasks

    def _filter_by_difficulty(
        self,
        difficulty_tolerance: Optional[str] = None
    ) -> List[str]:
        """
        Filter task queue by difficulty tolerance

        Args:
            difficulty_tolerance: "easy_only", "medium", or "all"

        Returns:
            List of task IDs matching tolerance
        """
        if difficulty_tolerance is None or difficulty_tolerance == "all":
            return list(self.task_queue)

        filtered = []
        for task_id in self.task_queue:
            task = self.task_inventory.get(task_id)
            if task is None:
                continue

            if difficulty_tolerance == "easy_only":
                if task.difficulty == TaskDifficulty.EASY:
                    filtered.append(task_id)
            elif difficulty_tolerance == "medium":
                if task.difficulty in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM]:
                    filtered.append(task_id)

        return filtered

    def mark_task_completed(self, task_id: str):
        """
        Mark a task as completed

        Args:
            task_id: ID of completed task
        """
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        self.total_tasks_completed += 1

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID

        Args:
            task_id: Task ID

        Returns:
            Task object or None if not found
        """
        return self.task_inventory.get(task_id)

    def get_task_statistics(self) -> Dict[str, Any]:
        """
        Return marketplace statistics

        Returns:
            Dictionary of statistics
        """
        # Count by type
        type_counts = {}
        for task in self.task_inventory.values():
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

        # Count by difficulty
        difficulty_counts = {}
        for task in self.task_inventory.values():
            difficulty = task.difficulty.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        # Calculate average price
        if self.task_inventory:
            avg_price = sum(t.price_per_label for t in self.task_inventory.values()) / len(self.task_inventory)
        else:
            avg_price = 0.0

        return {
            "total_tasks_created": self.total_tasks_created,
            "total_tasks_assigned": self.total_tasks_assigned,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_in_inventory": len(self.task_inventory),
            "available_tasks": len(self.task_queue),
            "tasks_by_type": type_counts,
            "tasks_by_difficulty": difficulty_counts,
            "avg_price_per_label": avg_price
        }

    def clear_completed_tasks(self):
        """
        Clear completed tasks from inventory to save memory
        (keep only tasks in queue)
        """
        # Get set of task IDs currently in queue
        queued_task_ids = set(self.task_queue)

        # Remove tasks not in queue
        tasks_to_remove = [
            task_id for task_id in self.task_inventory.keys()
            if task_id not in queued_task_ids
        ]

        for task_id in tasks_to_remove:
            del self.task_inventory[task_id]
