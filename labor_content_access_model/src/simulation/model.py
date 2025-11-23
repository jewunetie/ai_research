"""Main Mesa simulation model"""

import random
import uuid
from datetime import datetime
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from src.data.schemas import (
    UserType, CreatorSize, CompanySize, TaskType, ContentAccessSession
)
from src.agents.user_agent import UserAgentMesa
from src.agents.creator_agent import CreatorAgentMesa
from src.agents.company_agent import CompanyAgentMesa
from src.data.dataset_manager import DatasetManager
from src.marketplace.marketplace import MarketplaceEngine
from src.quality.quality_control import QualityControlSystem
from src.revenue.revenue_engine import RevenueEngine
from src.analytics.metrics_collector import MetricsCollector


class LaborContentAccessModel(Model):
    """
    Main simulation model for labor-for-content-access system

    Coordinates all agents and system components
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Set random seed for reproducibility
        random_seed = config.get("simulation", {}).get("random_seed", 42)
        random.seed(random_seed)
        self.random = random.Random(random_seed)

        # Initialize core systems
        self.dataset_manager = DatasetManager(config)
        self.marketplace = MarketplaceEngine(config)
        self.quality_control = QualityControlSystem(config, marketplace=self.marketplace)
        self.revenue_engine = RevenueEngine(config)
        self.metrics_collector = MetricsCollector()

        # Agent scheduler
        self.schedule = RandomActivation(self)

        # Agent lists (for easy access)
        self.users = []
        self.creators = []
        self.companies = []

        # Tracking
        self.current_step = 0
        self.sessions = []

        # Load datasets
        print("Loading datasets...")
        self.dataset_manager.load_datasets()

        # Create agents
        print("Creating agents...")
        self._create_agents()

        # Data collector for Mesa analytics
        self.datacollector = DataCollector(
            model_reporters={
                "Total Revenue": lambda m: m.revenue_engine.total_revenue,
                "Labor Revenue": lambda m: m.revenue_engine.revenue_by_source["labor"],
                "Ads Revenue": lambda m: m.revenue_engine.revenue_by_source["ads"],
                "Payment Revenue": lambda m: m.revenue_engine.revenue_by_source["payment"],
                "Total Sessions": lambda m: len(m.sessions),
                "Tasks in Marketplace": lambda m: len(m.marketplace.task_queue),
            }
        )

    def _create_agents(self):
        """Create all agents according to configuration"""
        agent_id = 0

        # Create users
        user_config = self.config.get("agents", {}).get("users", {})
        user_count = user_config.get("count", 1000)
        user_dist = user_config.get("distribution", {})

        user_types = []
        for user_type_str, percentage in user_dist.items():
            count = int(user_count * percentage)
            user_type = UserType(user_type_str)
            user_types.extend([user_type] * count)

        # Fill remaining to reach exact count
        while len(user_types) < user_count:
            user_types.append(UserType.BALANCED)

        for user_type in user_types:
            # Randomize behavioral parameters
            time_sens = self.random.choice(["low", "medium", "high"])
            diff_tol = self.random.choice(["easy_only", "medium", "all"])
            ad_tol = self.random.uniform(0, 1)

            user = UserAgentMesa(
                unique_id=agent_id,
                model=self,
                user_type=user_type,
                time_sensitivity=time_sens,
                task_difficulty_tolerance=diff_tol,
                ad_tolerance=ad_tol,
                privacy_concern="medium"
            )
            self.schedule.add(user)
            self.users.append(user)
            agent_id += 1

        print(f"  Created {len(self.users)} users")

        # Create creators
        creator_config = self.config.get("agents", {}).get("creators", {})
        creator_count = creator_config.get("count", 100)
        creator_dist = creator_config.get("distribution", {})
        size_params = creator_config.get("size_params", {})

        creator_sizes = []
        for size_str, percentage in creator_dist.items():
            count = int(creator_count * percentage)
            size = CreatorSize(size_str)
            creator_sizes.extend([size] * count)

        while len(creator_sizes) < creator_count:
            creator_sizes.append(CreatorSize.SMALL)

        for size in creator_sizes:
            params = size_params.get(size.value, {})
            monthly_range = params.get("monthly_visitors", [1000, 10000])
            cpm_range = params.get("current_cpm", [3.12, 5.00])

            monthly_visitors = self.random.randint(monthly_range[0], monthly_range[1])
            cpm = self.random.uniform(cpm_range[0], cpm_range[1])

            creator = CreatorAgentMesa(
                unique_id=agent_id,
                model=self,
                size=size,
                monthly_visitors=monthly_visitors,
                avg_visitors_per_day=monthly_visitors // 30,
                current_cpm=cpm,
                current_monthly_ad_revenue=(cpm / 1000) * monthly_visitors,
                content_value=self.random.uniform(5, 9),
                labor_task_count=creator_config.get("labor_task_count", 5)
            )
            self.schedule.add(creator)
            self.creators.append(creator)
            agent_id += 1

        print(f"  Created {len(self.creators)} creators")

        # Create companies
        company_config = self.config.get("agents", {}).get("companies", {})
        company_count = company_config.get("count", 10)
        company_dist = company_config.get("distribution", {})
        pricing = company_config.get("pricing", {})
        quality_reqs = company_config.get("quality_requirements", {})

        company_sizes = []
        for size_str, percentage in company_dist.items():
            count = int(company_count * percentage)
            size = CompanySize(size_str)
            company_sizes.extend([size] * count)

        while len(company_sizes) < company_count:
            company_sizes.append(CompanySize.STARTUP)

        for size in company_sizes:
            # Configure company needs
            task_types = [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_SENTIMENT]
            if size == CompanySize.ENTERPRISE:
                labels_needed = 100000
                budget = 10000.0
            elif size == CompanySize.MIDSIZE:
                labels_needed = 30000
                budget = 3000.0
            else:  # STARTUP
                labels_needed = 10000
                budget = 1000.0

            # Get quality requirement
            min_kappa = quality_reqs.get(size.value, 0.70)

            # Price per label by difficulty
            easy_price = self.random.uniform(pricing.get("easy", [0.01, 0.05])[0],
                                            pricing.get("easy", [0.01, 0.05])[1])
            medium_price = self.random.uniform(pricing.get("medium", [0.05, 0.15])[0],
                                              pricing.get("medium", [0.05, 0.15])[1])
            hard_price = self.random.uniform(pricing.get("hard", [0.15, 0.50])[0],
                                            pricing.get("hard", [0.15, 0.50])[1])

            from src.data.schemas import TaskDifficulty

            company = CompanyAgentMesa(
                unique_id=agent_id,
                model=self,
                size=size,
                task_types_needed=task_types,
                monthly_label_budget=budget,
                labels_needed_per_month=labels_needed,
                min_quality_kappa=min_kappa,
                max_price_per_label={
                    TaskDifficulty.EASY: easy_price,
                    TaskDifficulty.MEDIUM: medium_price,
                    TaskDifficulty.HARD: hard_price
                }
            )
            self.schedule.add(company)
            self.companies.append(company)
            agent_id += 1

        print(f"  Created {len(self.companies)} companies")

    def step(self):
        """
        Execute one step of the simulation
        """
        self.current_step += 1

        # Companies create tasks
        for company in self.companies:
            company.step()

        # Users make decisions and interact
        # Only activate a subset of users per step to simulate realistic traffic
        active_users = self.random.sample(self.users, min(100, len(self.users)))
        for user in active_users:
            user.step()

        # Creators update stats
        for creator in self.creators:
            creator.step()

        # Collect data
        self.datacollector.collect(self)

        # Periodic cleanup
        if self.current_step % 100 == 0:
            self.marketplace.clear_completed_tasks()
            print(f"  Step {self.current_step}: "
                  f"{len(self.sessions)} sessions, "
                  f"{self.marketplace.total_tasks_assigned} tasks assigned, "
                  f"${self.revenue_engine.total_revenue:.2f} revenue")

    def record_session(
        self,
        user_id: str,
        creator_id: str,
        choice: str,
        tasks_assigned=None,
        labels_submitted=None,
        total_task_time=0.0,
        revenue_to_creator=0.0,
        revenue_to_platform=0.0
    ):
        """
        Record a content access session

        Args:
            user_id: User ID
            creator_id: Creator ID
            choice: User's choice ("ads", "payment", "labor")
            tasks_assigned: List of task IDs (optional)
            labels_submitted: List of label IDs (optional)
            total_task_time: Total time spent on tasks (optional)
            revenue_to_creator: Revenue going to creator
            revenue_to_platform: Revenue going to platform
        """
        session = ContentAccessSession(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            creator_id=creator_id,
            choice=choice,
            decision_time_ms=self.random.uniform(500, 2000),  # Simulated decision time
            tasks_assigned=tasks_assigned,
            labels_submitted=labels_submitted,
            total_task_time_seconds=total_task_time,
            revenue_to_creator=revenue_to_creator,
            revenue_to_platform=revenue_to_platform,
            revenue_source=choice,
            access_granted=True
        )

        self.sessions.append(session)
        self.metrics_collector.record_session(session)

    def run_simulation(self, num_steps: int):
        """
        Run the simulation for specified number of steps

        Args:
            num_steps: Number of steps to run
        """
        print(f"\nRunning simulation for {num_steps} steps...")
        for i in range(num_steps):
            self.step()

        print(f"\nSimulation complete!")
        print(f"Total sessions: {len(self.sessions)}")
        print(f"Total revenue: ${self.revenue_engine.total_revenue:.2f}")

    def get_summary_report(self):
        """
        Generate comprehensive summary report

        Returns:
            Dictionary of summary statistics
        """
        metrics_report = self.metrics_collector.generate_report()
        quality_metrics = self.quality_control.get_quality_metrics()
        revenue_stats = self.revenue_engine.get_summary_statistics()
        marketplace_stats = self.marketplace.get_task_statistics()

        return {
            "simulation": {
                "total_steps": self.current_step,
                "total_sessions": len(self.sessions),
                "num_users": len(self.users),
                "num_creators": len(self.creators),
                "num_companies": len(self.companies)
            },
            "metrics": metrics_report,
            "quality": quality_metrics,
            "revenue": revenue_stats,
            "marketplace": marketplace_stats
        }
