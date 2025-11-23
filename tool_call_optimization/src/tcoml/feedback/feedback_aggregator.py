from typing import List, Dict, Any
from collections import Counter
from pydantic import BaseModel
from ..llm.base_client import BaseLLMClient, Message
from .feedback_generator import Feedback


class AggregatedFeedback(BaseModel):
    """Aggregated feedback across multiple traces"""
    total_traces: int
    successful_traces: int
    failed_traces: int
    common_failure_patterns: List[str]
    common_success_patterns: List[str]
    top_suggestions: List[str]
    summary: str


class FeedbackAggregator:
    """
    Aggregates feedback from multiple traces to identify patterns.

    This helps the meta-prompter focus on systemic issues rather
    than task-specific problems.
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def aggregate(self, feedbacks: List[Feedback]) -> AggregatedFeedback:
        """
        Aggregate multiple feedback items into patterns.

        Args:
            feedbacks: List of individual feedback items

        Returns:
            Aggregated feedback with common patterns
        """
        if not feedbacks:
            return AggregatedFeedback(
                total_traces=0,
                successful_traces=0,
                failed_traces=0,
                common_failure_patterns=[],
                common_success_patterns=[],
                top_suggestions=[],
                summary="No feedback to aggregate"
            )

        # Basic statistics
        total = len(feedbacks)
        successful = sum(1 for f in feedbacks if f.success)
        failed = total - successful

        # Collect all suggestions
        all_suggestions = []
        for f in feedbacks:
            all_suggestions.extend(f.suggestions)

        # Find most common suggestions (simple frequency)
        suggestion_counts = Counter(all_suggestions)
        top_suggestions = [s for s, count in suggestion_counts.most_common(10)]

        # Analyze failure patterns
        failure_feedbacks = [f for f in feedbacks if not f.success]
        failure_patterns = self._identify_patterns(
            [f.what_went_wrong for f in failure_feedbacks],
            "failure"
        )

        # Analyze success patterns
        success_feedbacks = [f for f in feedbacks if f.success]
        success_patterns = self._identify_patterns(
            [f.what_went_right for f in success_feedbacks],
            "success"
        )

        # Generate summary
        summary = self._generate_summary(
            total, successful, failed,
            failure_patterns, success_patterns, top_suggestions
        )

        return AggregatedFeedback(
            total_traces=total,
            successful_traces=successful,
            failed_traces=failed,
            common_failure_patterns=failure_patterns,
            common_success_patterns=success_patterns,
            top_suggestions=top_suggestions,
            summary=summary
        )

    def _identify_patterns(self, texts: List[str], pattern_type: str) -> List[str]:
        """Use LLM to identify common patterns in feedback texts"""
        if not texts:
            return []

        # Combine all texts
        combined = "\n\n".join([f"- {t}" for t in texts if t])

        if not combined:
            return []

        prompt = f"""Analyze the following {pattern_type} descriptions and identify 3-5 common patterns.

{combined}

Extract the most common {pattern_type} patterns. Be specific and concise.
Return as a simple list:
- Pattern 1
- Pattern 2
- ...
"""

        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=500
        )

        # Parse patterns from response
        patterns = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                pattern = line.lstrip("-•* ").strip()
                if pattern:
                    patterns.append(pattern)

        return patterns[:5]  # Top 5

    def _generate_summary(
        self,
        total: int,
        successful: int,
        failed: int,
        failure_patterns: List[str],
        success_patterns: List[str],
        top_suggestions: List[str]
    ) -> str:
        """Generate natural language summary"""

        success_rate = successful / total if total > 0 else 0

        summary_parts = [
            f"Analyzed {total} task executions ({successful} successful, {failed} failed).",
            f"Success rate: {success_rate:.1%}",
        ]

        if failure_patterns:
            summary_parts.append("\nCommon failure patterns:")
            summary_parts.extend([f"- {p}" for p in failure_patterns[:3]])

        if success_patterns:
            summary_parts.append("\nCommon success patterns:")
            summary_parts.extend([f"- {p}" for p in success_patterns[:3]])

        if top_suggestions:
            summary_parts.append("\nTop suggestions for improvement:")
            summary_parts.extend([f"- {s}" for s in top_suggestions[:3]])

        return "\n".join(summary_parts)
