"""Revenue calculation and distribution engine"""

from typing import Dict, Any, Optional
from collections import defaultdict


class RevenueEngine:
    """
    Calculates and distributes revenue between platform, creators, and users
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Revenue split configuration
        revenue_config = config.get("revenue", {})
        self.platform_share = revenue_config.get("platform_share", 0.20)  # 20%
        self.creator_share = revenue_config.get("creator_share", 0.70)    # 70%
        self.user_share = revenue_config.get("user_share", 0.10)          # 10% (optional)

        # Tracking
        self.total_revenue = 0.0
        self.revenue_by_creator: Dict[str, float] = defaultdict(float)
        self.revenue_by_source: Dict[str, float] = {
            "labor": 0.0,
            "ads": 0.0,
            "payment": 0.0
        }

        # Additional tracking
        self.platform_revenue = 0.0
        self.creator_revenue = 0.0
        self.user_revenue = 0.0

    def distribute(
        self,
        revenue: float,
        creator_id: str,
        source: str = "labor",
        user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Distribute revenue according to configured splits

        Args:
            revenue: Total revenue to distribute
            creator_id: ID of creator
            source: Revenue source ("labor", "ads", "payment")
            user_id: ID of user (optional, for user rewards)

        Returns:
            Dictionary with split amounts: {
                'platform': amount,
                'creator': amount,
                'user': amount
            }
        """
        self.total_revenue += revenue
        self.revenue_by_source[source] += revenue

        if source == "labor":
            # Split labor revenue
            platform_cut = revenue * self.platform_share
            creator_cut = revenue * self.creator_share
            user_cut = revenue * self.user_share

            # Record
            self.revenue_by_creator[creator_id] += creator_cut
            self.platform_revenue += platform_cut
            self.creator_revenue += creator_cut
            self.user_revenue += user_cut

            return {
                'platform': platform_cut,
                'creator': creator_cut,
                'user': user_cut,
                'total': revenue
            }

        elif source == "ads":
            # All ad revenue to creator (platform already took cut via CPM)
            self.revenue_by_creator[creator_id] += revenue
            self.creator_revenue += revenue

            return {
                'platform': 0.0,
                'creator': revenue,
                'user': 0.0,
                'total': revenue
            }

        else:  # payment
            # Payment revenue to creator (minus platform fee)
            platform_cut = revenue * 0.10  # Typical platform fee
            creator_cut = revenue * 0.90

            self.revenue_by_creator[creator_id] += creator_cut
            self.platform_revenue += platform_cut
            self.creator_revenue += creator_cut

            return {
                'platform': platform_cut,
                'creator': creator_cut,
                'user': 0.0,
                'total': revenue
            }

    def calculate_cpm_equivalent(
        self,
        creator_id: str,
        impressions: int,
        source: str = "labor"
    ) -> float:
        """
        Calculate effective CPM from revenue

        CPM = (Revenue / Impressions) * 1000

        Args:
            creator_id: Creator ID
            impressions: Number of impressions
            source: Revenue source to calculate CPM for

        Returns:
            CPM value
        """
        revenue = self.revenue_by_creator.get(creator_id, 0.0)
        if impressions == 0:
            return 0.0
        return (revenue / impressions) * 1000

    def compare_to_ads(
        self,
        creator_id: str,
        ad_cpm: float,
        impressions: int
    ) -> Dict[str, Any]:
        """
        Compare labor revenue to what ads would have generated

        Args:
            creator_id: Creator ID
            ad_cpm: Advertising CPM benchmark
            impressions: Number of impressions

        Returns:
            Dictionary with comparison metrics
        """
        labor_revenue = self.revenue_by_creator.get(creator_id, 0.0)
        labor_cpm = self.calculate_cpm_equivalent(creator_id, impressions)

        ad_revenue_equivalent = (ad_cpm / 1000) * impressions

        difference = labor_revenue - ad_revenue_equivalent
        if ad_revenue_equivalent > 0:
            difference_pct = (difference / ad_revenue_equivalent) * 100
        else:
            difference_pct = 0.0

        viable = labor_revenue >= ad_revenue_equivalent * 0.9  # Within 10%

        return {
            "labor_revenue": labor_revenue,
            "labor_cpm": labor_cpm,
            "ad_revenue_equivalent": ad_revenue_equivalent,
            "ad_cpm": ad_cpm,
            "difference": difference,
            "difference_pct": difference_pct,
            "viable": viable,
            "impressions": impressions
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all revenue

        Returns:
            Dictionary of revenue statistics
        """
        num_creators = len(self.revenue_by_creator)

        return {
            "total_revenue": self.total_revenue,
            "revenue_by_source": self.revenue_by_source.copy(),
            "platform_revenue": self.platform_revenue,
            "creator_revenue": self.creator_revenue,
            "user_revenue": self.user_revenue,
            "num_creators_with_revenue": num_creators,
            "avg_revenue_per_creator": (
                self.creator_revenue / num_creators if num_creators > 0 else 0.0
            ),
            "labor_percentage": (
                (self.revenue_by_source["labor"] / self.total_revenue * 100)
                if self.total_revenue > 0 else 0.0
            ),
            "ads_percentage": (
                (self.revenue_by_source["ads"] / self.total_revenue * 100)
                if self.total_revenue > 0 else 0.0
            ),
            "payment_percentage": (
                (self.revenue_by_source["payment"] / self.total_revenue * 100)
                if self.total_revenue > 0 else 0.0
            )
        }

    def get_creator_revenue(self, creator_id: str) -> float:
        """
        Get total revenue for a specific creator

        Args:
            creator_id: Creator ID

        Returns:
            Total revenue
        """
        return self.revenue_by_creator.get(creator_id, 0.0)
