"""Metrics collection and analysis for simulation results"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, Any, List
from pathlib import Path
from src.data.schemas import ContentAccessSession


class MetricsCollector:
    """
    Collects and reports simulation metrics
    """

    def __init__(self):
        self.sessions: List[ContentAccessSession] = []
        self.user_choices: List[str] = []

    def record_session(self, session: ContentAccessSession):
        """
        Record a content access session

        Args:
            session: ContentAccessSession object
        """
        self.sessions.append(session)
        self.user_choices.append(session.choice)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report

        Returns:
            Dictionary of metrics
        """
        if not self.sessions:
            return {
                "error": "No sessions recorded",
                "total_sessions": 0
            }

        # Choice distribution
        choice_dist = Counter(self.user_choices)
        total = len(self.user_choices)

        # Convert sessions to DataFrame for analysis
        df = pd.DataFrame([s.model_dump() for s in self.sessions])

        report = {
            "summary": {
                "total_sessions": total,
                "choice_distribution": {
                    "labor_count": choice_dist.get("labor", 0),
                    "ads_count": choice_dist.get("ads", 0),
                    "payment_count": choice_dist.get("payment", 0),
                    "labor_pct": (choice_dist.get("labor", 0) / total * 100) if total > 0 else 0,
                    "ads_pct": (choice_dist.get("ads", 0) / total * 100) if total > 0 else 0,
                    "payment_pct": (choice_dist.get("payment", 0) / total * 100) if total > 0 else 0
                }
            },
            "revenue": {
                "total": float(df["revenue_to_creator"].sum()),
                "by_source": df.groupby("revenue_source")["revenue_to_creator"].sum().to_dict() if "revenue_source" in df.columns else {},
                "avg_per_session": float(df["revenue_to_creator"].mean()),
                "total_platform_revenue": float(df["revenue_to_platform"].sum())
            },
            "timing": {
                "avg_task_time": float(df[df["choice"] == "labor"]["total_task_time_seconds"].mean()) if "total_task_time_seconds" in df.columns else 0.0,
                "avg_decision_time": float(df["decision_time_ms"].mean()) if "decision_time_ms" in df.columns else 0.0
            },
            "quality": {
                "avg_label_quality": float(df[df["choice"] == "labor"]["avg_label_quality"].mean()) if "avg_label_quality" in df.columns else 0.0
            }
        }

        return report

    def visualize_results(self, output_dir: str = "./results"):
        """
        Generate visualization plots

        Args:
            output_dir: Directory to save plots
        """
        if not self.sessions:
            print("No sessions to visualize")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([s.model_dump() for s in self.sessions])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Choice distribution pie chart
        choice_counts = df["choice"].value_counts()
        axes[0, 0].pie(choice_counts.values, labels=choice_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title("User Choice Distribution")

        # 2. Revenue by source
        if "revenue_source" in df.columns:
            revenue_by_source = df.groupby("revenue_source")["revenue_to_creator"].sum()
            if len(revenue_by_source) > 0:
                axes[0, 1].bar(revenue_by_source.index, revenue_by_source.values)
                axes[0, 1].set_title("Revenue by Source")
                axes[0, 1].set_ylabel("Revenue ($)")
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, "No revenue data available",
                               ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, "No revenue source data",
                           ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Task time distribution
        labor_sessions = df[df["choice"] == "labor"]
        if len(labor_sessions) > 0 and "total_task_time_seconds" in labor_sessions.columns:
            task_times = labor_sessions["total_task_time_seconds"].dropna()
            if len(task_times) > 0:
                axes[1, 0].hist(task_times, bins=30, edgecolor='black')
                axes[1, 0].set_title("Task Completion Time Distribution")
                axes[1, 0].set_xlabel("Time (seconds)")
                axes[1, 0].set_ylabel("Frequency")
            else:
                axes[1, 0].text(0.5, 0.5, "No task time data",
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, "No labor sessions",
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Quality metrics over time
        if len(labor_sessions) > 0 and "avg_label_quality" in labor_sessions.columns:
            quality_data = labor_sessions[labor_sessions["avg_label_quality"].notna()]
            if len(quality_data) > 0:
                axes[1, 1].plot(quality_data.index, quality_data["avg_label_quality"], marker='o', markersize=2)
                axes[1, 1].set_title("Label Quality Over Time")
                axes[1, 1].set_ylabel("Fleiss' Kappa")
                axes[1, 1].set_xlabel("Session Index")
                axes[1, 1].axhline(y=0.75, color='r', linestyle='--', label='Excellent (>0.75)')
                axes[1, 1].axhline(y=0.60, color='orange', linestyle='--', label='Good (>0.60)')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, "No quality data available",
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, "No quality data",
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(output_path / "simulation_results.png", dpi=300)
        plt.close()

        print(f"Visualizations saved to {output_path}/simulation_results.png")

    def export_to_csv(self, output_dir: str = "./results"):
        """
        Export sessions to CSV

        Args:
            output_dir: Directory to save CSV
        """
        if not self.sessions:
            print("No sessions to export")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([s.model_dump() for s in self.sessions])
        csv_path = output_path / "sessions.csv"
        df.to_csv(csv_path, index=False)

        print(f"Sessions exported to {csv_path}")
