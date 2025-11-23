"""Quality control system for validating labels and computing consensus"""

import numpy as np
from collections import Counter
from typing import Dict, Any, List, Optional
from src.data.schemas import Label, Task


class QualityControlSystem:
    """
    Validates labels and computes quality metrics using consensus algorithms
    """

    def __init__(self, config: Dict[str, Any], marketplace=None):
        self.config = config
        self.redundancy_factor = config.get("marketplace", {}).get("redundancy_factor", 3)
        self.marketplace = marketplace  # Reference to get tasks

        # Storage for multi-label consensus
        self.task_labels: Dict[str, List[Label]] = {}
        self.tasks: Dict[str, Task] = {}  # Cache of tasks for lookup

        # Statistics
        self.total_labels_processed = 0
        self.total_revenue_paid = 0.0
        self.quality_scores = []

    def get_task(self, task_id: str) -> Task:
        """
        Retrieve task by ID from cache or marketplace

        Args:
            task_id: Task ID

        Returns:
            Task object

        Raises:
            ValueError: If task not found
        """
        if task_id in self.tasks:
            return self.tasks[task_id]
        elif self.marketplace:
            task = self.marketplace.task_inventory.get(task_id)
            if task:
                self.tasks[task_id] = task
                return task
        raise ValueError(f"Task {task_id} not found")

    def process_labels(self, labels: List[Label]) -> float:
        """
        Process submitted labels and calculate revenue

        Args:
            labels: List of label submissions

        Returns:
            Total revenue generated from these labels
        """
        total_revenue = 0.0

        for label in labels:
            self.total_labels_processed += 1

            # Store label for consensus
            if label.task_id not in self.task_labels:
                self.task_labels[label.task_id] = []
            self.task_labels[label.task_id].append(label)

            # Check if we have enough labels for consensus
            if len(self.task_labels[label.task_id]) >= self.redundancy_factor:
                revenue = self.compute_consensus_and_pay(label.task_id)
                total_revenue += revenue

                # Mark task as completed in marketplace
                if self.marketplace:
                    self.marketplace.mark_task_completed(label.task_id)

        return total_revenue

    def compute_consensus_and_pay(self, task_id: str) -> float:
        """
        Compute consensus label and pay if quality meets threshold

        Args:
            task_id: Task ID

        Returns:
            Total payment for this task
        """
        labels = self.task_labels[task_id]

        # Get task and ground truth
        task = self.get_task(task_id)

        # Majority vote
        submitted_labels = [l.submitted_label for l in labels]
        consensus_label = Counter(submitted_labels).most_common(1)[0][0]

        # Compute agreement (Fleiss' kappa)
        kappa = self.compute_fleiss_kappa(submitted_labels)
        self.quality_scores.append(kappa)

        # Check accuracy against ground truth
        accuracy = 1.0 if consensus_label == task.ground_truth_label else 0.0

        # Update label correctness
        for label in labels:
            label.is_correct = (label.submitted_label == task.ground_truth_label)
            label.quality_score = kappa

        # Determine payment
        if kappa >= task.quality_requirement:
            # Quality meets threshold
            base_payment = task.price_per_label * len(labels)

            # Bonus for exceptional quality
            # Get bonus multiplier from config (default 1.5)
            bonus_multiplier = self.config.get("marketplace", {}).get("quality_bonus_multiplier", 1.5)
            if kappa > 0.75:
                payment = base_payment * bonus_multiplier
            else:
                payment = base_payment

            self.total_revenue_paid += payment
            return payment
        else:
            # Quality too low, no payment
            return 0.0

    def compute_fleiss_kappa(self, labels: List[Any]) -> float:
        """
        Compute Fleiss' kappa for multiple annotators

        This is a simplified approximation. For a real implementation,
        use statsmodels.stats.inter_rater.fleiss_kappa

        Args:
            labels: List of submitted labels

        Returns:
            Fleiss' kappa score (0-1)
        """
        if len(labels) < 2:
            return 1.0

        # Simple agreement rate
        most_common = Counter(labels).most_common(1)[0][1]
        agreement = most_common / len(labels)

        # Rough kappa approximation (simplified)
        # Real: use statsmodels.stats.inter_rater.fleiss_kappa
        num_categories = len(set(labels))

        # Handle perfect agreement (all labels identical)
        if num_categories == 1:
            return 1.0

        expected_agreement = 1.0 / num_categories
        kappa = (agreement - expected_agreement) / (1 - expected_agreement)

        return max(0.0, min(1.0, kappa))

    def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Return overall quality metrics

        Returns:
            Dictionary of quality metrics
        """
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
            "avg_fleiss_kappa": float(np.mean(all_kappas)) if all_kappas else 0.0,
            "avg_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
            "tasks_with_consensus": len(all_kappas),
            "total_labels_processed": self.total_labels_processed,
            "total_revenue_paid": self.total_revenue_paid,
            "kappa_distribution": {
                "excellent (>0.75)": sum(1 for k in all_kappas if k > 0.75),
                "good (0.60-0.75)": sum(1 for k in all_kappas if 0.60 <= k <= 0.75),
                "fair (0.40-0.60)": sum(1 for k in all_kappas if 0.40 <= k < 0.60),
                "poor (<0.40)": sum(1 for k in all_kappas if k < 0.40)
            }
        }
