"""Cost tracking utilities for experiments."""

from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class CostEstimate:
    """Cost estimate for API usage."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0
    calls_made: int = 0

    def add_call(self, input_tokens: int, output_tokens: int, cost: float):
        """Add a single API call to the estimate."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.estimated_cost += cost
        self.calls_made += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": self.estimated_cost,
            "calls_made": self.calls_made,
        }


class CostTracker:
    """Track API costs during experiments."""

    def __init__(self, pricing: Dict[str, Dict[str, float]], model_name: str):
        """
        Initialize cost tracker.

        Args:
            pricing: Model pricing dictionary
            model_name: Name of model being used
        """
        self.pricing = pricing
        self.model_name = model_name
        self.estimate = CostEstimate()

        # Get pricing for this model
        if model_name in pricing:
            self.input_price_per_1k = pricing[model_name]["input_per_1k"]
            self.output_price_per_1k = pricing[model_name]["output_per_1k"]
        else:
            # Default to GPT-4o pricing if model not found
            print(f"⚠  Warning: No pricing info for {model_name}, using GPT-4o defaults")
            self.input_price_per_1k = 0.0025
            self.output_price_per_1k = 0.01

    def track_call(self, input_tokens: int, output_tokens: int):
        """
        Track a single API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # Calculate cost
        input_cost = (input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_price_per_1k
        total_cost = input_cost + output_cost

        # Add to estimate
        self.estimate.add_call(input_tokens, output_tokens, total_cost)

    def get_summary(self) -> str:
        """Get human-readable cost summary."""
        return (
            f"API Usage:\n"
            f"  Calls: {self.estimate.calls_made:,}\n"
            f"  Input tokens: {self.estimate.total_input_tokens:,}\n"
            f"  Output tokens: {self.estimate.total_output_tokens:,}\n"
            f"  Estimated cost: ${self.estimate.estimated_cost:.2f}"
        )

    def estimate_remaining(self, completed: int, total: int) -> str:
        """
        Estimate remaining cost based on progress.

        Args:
            completed: Number of items completed
            total: Total number of items

        Returns:
            Formatted string with estimate
        """
        if completed == 0:
            return "Estimating..."

        # Calculate average cost per item
        avg_cost_per_item = self.estimate.estimated_cost / completed

        # Estimate remaining
        remaining = total - completed
        estimated_remaining = avg_cost_per_item * remaining
        estimated_total = self.estimate.estimated_cost + estimated_remaining

        return (
            f"Estimated remaining: ${estimated_remaining:.2f} "
            f"(Total: ${estimated_total:.2f})"
        )
