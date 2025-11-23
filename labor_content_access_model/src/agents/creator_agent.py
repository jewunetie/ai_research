"""Creator agent implementation for agent-based modeling"""

from mesa import Agent
from src.data.schemas import CreatorAgent, CreatorSize


class CreatorAgentMesa(Agent):
    """
    Mesa-based creator agent

    Represents a content creator/website offering content access
    """

    def __init__(self, unique_id, model, size: CreatorSize, **params):
        super().__init__(unique_id, model)
        self.creator_data = CreatorAgent(
            creator_id=str(unique_id),
            size=size,
            **params
        )

    def step(self):
        """
        Track daily visitors and revenue

        In the simulation, user agents initiate interactions,
        so creator agents primarily track state
        """
        # Increment total access events (tracked by user interactions)
        self.creator_data.total_access_events = (
            self.creator_data.labor_access_count +
            self.creator_data.ad_access_count +
            self.creator_data.payment_access_count
        )

    def get_stats(self):
        """
        Get creator statistics

        Returns:
            Dictionary of creator stats
        """
        total_access = self.creator_data.total_access_events

        return {
            "creator_id": self.creator_data.creator_id,
            "size": self.creator_data.size.value,
            "total_access_events": total_access,
            "labor_access_count": self.creator_data.labor_access_count,
            "ad_access_count": self.creator_data.ad_access_count,
            "payment_access_count": self.creator_data.payment_access_count,
            "labor_percentage": (
                (self.creator_data.labor_access_count / total_access * 100)
                if total_access > 0 else 0.0
            ),
            "total_labor_revenue": self.creator_data.total_labor_revenue,
            "total_ad_revenue": self.creator_data.total_ad_revenue,
            "total_payment_revenue": self.creator_data.total_payment_revenue,
            "total_revenue": (
                self.creator_data.total_labor_revenue +
                self.creator_data.total_ad_revenue +
                self.creator_data.total_payment_revenue
            )
        }
