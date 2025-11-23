"""User agent implementation for agent-based modeling"""

import random
import uuid
from datetime import datetime
from mesa import Agent
from src.data.schemas import UserAgent, UserType, TaskDifficulty, Label, Task


class UserAgentMesa(Agent):
    """
    Mesa-based user agent for agent-based modeling

    Simulates user behavior in making access decisions and completing tasks
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
        if not hasattr(self.model, 'creators') or not self.model.creators:
            return

        creator = self.random.choice(self.model.creators)

        # Make access choice decision
        choice = self.decide_access_method(creator)

        # Execute chosen method
        if choice == "labor":
            self.complete_labeling_tasks(creator)
        elif choice == "ads":
            self.view_ads(creator)
        else:  # payment
            self.pay_subscription(creator)

    def decide_access_method(self, creator) -> str:
        """
        Utility-based decision making

        Utility function:
        U(ads) = content_value - (ad_time * ad_tolerance_factor)
        U(labor) = content_value - (task_time * difficulty_sensitivity)
        U(payment) = content_value - (price * payment_sensitivity)

        Choose option with highest utility

        Args:
            creator: CreatorAgentMesa instance

        Returns:
            Choice: "ads", "payment", or "labor"
        """
        content_value = creator.creator_data.content_value

        # Calculate utilities
        # Ads: 30 seconds of viewing
        u_ads = content_value - (30 * (1 - self.user_data.ad_tolerance))

        # Labor: task_count * avg_time_per_task
        avg_task_time = creator.creator_data.labor_task_count * 15  # 15 sec per task
        difficulty_factor = 1.0  # Simplified, would check actual task difficulty
        u_labor = content_value - (avg_task_time * difficulty_factor * self.user_data.weight_time / 60)

        # Payment: typical $5 payment
        payment_cost = 5.0
        u_payment = content_value - (payment_cost * self.user_data.weight_payment)

        # Add some randomness based on user type
        if self.user_data.user_type == UserType.TASK_AVOIDER:
            u_labor *= 0.5  # Heavily penalize labor
        elif self.user_data.user_type == UserType.TASK_PREFERER:
            u_labor *= 1.5  # Prefer labor
        elif self.user_data.user_type == UserType.PAYMENT_PREFERER:
            u_payment *= 1.3  # Prefer payment

        # Choose max utility
        utilities = {"ads": u_ads, "labor": u_labor, "payment": u_payment}
        return max(utilities, key=utilities.get)

    def complete_labeling_tasks(self, creator):
        """
        Simulate completing labeling tasks

        Args:
            creator: CreatorAgentMesa instance
        """
        # Get tasks from marketplace
        tasks = self.model.marketplace.assign_tasks(
            user_id=self.user_data.user_id,
            count=creator.creator_data.labor_task_count,
            difficulty_tolerance=self.user_data.task_difficulty_tolerance
        )

        if not tasks:
            # No tasks available, fall back to ads
            self.view_ads(creator)
            return

        # Complete each task
        labels = []
        total_time = 0.0
        for task in tasks:
            label = self.simulate_label(task)
            labels.append(label)
            total_time += label.time_taken_seconds

        # Submit to quality control
        revenue = self.model.quality_control.process_labels(labels)

        # Distribute revenue
        distribution = self.model.revenue_engine.distribute(
            revenue=revenue,
            creator_id=creator.creator_data.creator_id,
            source="labor",
            user_id=self.user_data.user_id
        )

        # Record session
        self.model.record_session(
            user_id=self.user_data.user_id,
            creator_id=creator.creator_data.creator_id,
            choice="labor",
            tasks_assigned=[t.task_id for t in tasks],
            labels_submitted=[l.label_id for l in labels],
            total_task_time=total_time,
            revenue_to_creator=distribution['creator'],
            revenue_to_platform=distribution['platform']
        )

        # Update user stats
        self.user_data.tasks_completed += len(tasks)
        self.user_data.total_time_spent += total_time

        # Update creator stats
        creator.creator_data.labor_access_count += 1
        creator.creator_data.total_labor_revenue += distribution['creator']

    def simulate_label(self, task: Task) -> Label:
        """
        Simulate user labeling a task
        Returns label with accuracy based on user's skill and task difficulty

        Args:
            task: Task to label

        Returns:
            Label object
        """
        # Get user's accuracy for this difficulty
        accuracy = self.user_data.base_accuracy
        if self.user_data.difficulty_penalty:
            penalty = self.user_data.difficulty_penalty.get(task.difficulty, 0.0)
            accuracy -= penalty

        # Determine if user gets it correct
        is_correct = random.random() < accuracy

        if is_correct:
            submitted_label = task.ground_truth_label
        else:
            # Submit wrong answer
            possible_labels = [l for l in task.possible_labels if l != str(task.ground_truth_label)]
            if possible_labels:
                submitted_label = random.choice(possible_labels)
            else:
                submitted_label = task.ground_truth_label  # Fallback

        # Simulate time taken (with some variance)
        time_taken = task.estimated_time_seconds * random.uniform(0.8, 1.2)

        label = Label(
            label_id=str(uuid.uuid4()),
            task_id=task.task_id,
            user_id=self.user_data.user_id,
            submitted_label=submitted_label,
            confidence=None,
            time_taken_seconds=time_taken,
            submitted_at=datetime.now()
        )

        return label

    def view_ads(self, creator):
        """
        Simulate viewing advertisements

        Args:
            creator: CreatorAgentMesa instance
        """
        # Calculate ad revenue (CPM-based)
        ad_revenue = (creator.creator_data.current_cpm / 1000)

        # Record session
        self.model.record_session(
            user_id=self.user_data.user_id,
            creator_id=creator.creator_data.creator_id,
            choice="ads",
            revenue_to_creator=ad_revenue,
            revenue_to_platform=0.0
        )

        # Update creator stats
        creator.creator_data.ad_access_count += 1
        creator.creator_data.total_ad_revenue += ad_revenue

        # Distribute revenue
        self.model.revenue_engine.distribute(
            revenue=ad_revenue,
            creator_id=creator.creator_data.creator_id,
            source="ads"
        )

    def pay_subscription(self, creator):
        """
        Simulate paying subscription

        Args:
            creator: CreatorAgentMesa instance
        """
        payment_amount = 5.0  # $5 typical payment

        # Record session
        self.model.record_session(
            user_id=self.user_data.user_id,
            creator_id=creator.creator_data.creator_id,
            choice="payment",
            revenue_to_creator=payment_amount * 0.9,  # 90% to creator
            revenue_to_platform=payment_amount * 0.1  # 10% platform fee
        )

        # Update creator stats
        creator.creator_data.payment_access_count += 1
        creator.creator_data.total_payment_revenue += payment_amount * 0.9

        # Distribute revenue
        self.model.revenue_engine.distribute(
            revenue=payment_amount,
            creator_id=creator.creator_data.creator_id,
            source="payment"
        )
