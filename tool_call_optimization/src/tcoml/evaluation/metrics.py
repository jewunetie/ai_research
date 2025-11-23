import numpy as np
from typing import List, Dict, Any
from ..tracing.trace_models import ExecutionTrace, TraceDataset


class EvaluationMetrics:
    """Comprehensive evaluation metrics for agent performance"""

    @staticmethod
    def success_rate(traces: List[ExecutionTrace]) -> float:
        """Calculate task success rate"""
        if not traces:
            return 0.0
        return sum(1 for t in traces if t.task_success) / len(traces)

    @staticmethod
    def avg_tool_calls(traces: List[ExecutionTrace]) -> float:
        """Average number of tool calls per task"""
        if not traces:
            return 0.0
        return sum(len(t.tool_calls) for t in traces) / len(traces)

    @staticmethod
    def avg_successful_tool_calls(traces: List[ExecutionTrace]) -> float:
        """Average number of successful tool calls"""
        if not traces:
            return 0.0
        successful = sum(
            sum(1 for tc in t.tool_calls if tc.success)
            for t in traces
        )
        return successful / len(traces)

    @staticmethod
    def tool_call_efficiency(traces: List[ExecutionTrace]) -> float:
        """
        Ratio of successful tool calls to total tool calls.
        Higher is better (less wasted calls).
        """
        total_calls = sum(len(t.tool_calls) for t in traces)
        if total_calls == 0:
            return 0.0

        successful_calls = sum(
            sum(1 for tc in t.tool_calls if tc.success)
            for t in traces
        )
        return successful_calls / total_calls

    @staticmethod
    def first_attempt_success_rate(traces: List[ExecutionTrace]) -> float:
        """
        Percentage of tasks solved without any failed tool calls.
        Measures how often the agent gets it right immediately.
        """
        if not traces:
            return 0.0

        first_attempt = sum(
            1 for t in traces
            if t.task_success and all(tc.success for tc in t.tool_calls)
        )
        return first_attempt / len(traces)

    @staticmethod
    def error_recovery_rate(traces: List[ExecutionTrace]) -> float:
        """
        Percentage of tasks that succeeded despite having failed tool calls.
        Measures agent's ability to recover from errors.
        """
        if not traces:
            return 0.0

        recovered = sum(
            1 for t in traces
            if t.task_success and any(not tc.success for tc in t.tool_calls)
        )

        # Denominator: tasks that had at least one error
        tasks_with_errors = sum(
            1 for t in traces
            if any(not tc.success for tc in t.tool_calls)
        )

        if tasks_with_errors == 0:
            return 0.0

        return recovered / tasks_with_errors

    @staticmethod
    def avg_execution_time(traces: List[ExecutionTrace]) -> float:
        """Average execution time in milliseconds"""
        if not traces:
            return 0.0
        return sum(t.total_time_ms for t in traces) / len(traces)

    @staticmethod
    def bootstrap_confidence_interval(
        traces: List[ExecutionTrace],
        metric_fn: callable,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> tuple:
        """
        Calculate bootstrap confidence interval for a metric.

        Returns:
            (metric_value, lower_bound, upper_bound)
        """
        metric_value = metric_fn(traces)

        # Bootstrap resampling
        bootstrap_values = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(traces, size=len(traces), replace=True)
            bootstrap_values.append(metric_fn(list(sample)))

        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

        return metric_value, lower, upper

    @staticmethod
    def compare_performance(
        baseline_traces: List[ExecutionTrace],
        optimized_traces: List[ExecutionTrace],
        metric_fn: callable = None
    ) -> Dict[str, Any]:
        """
        Compare performance between baseline and optimized.

        Returns statistical significance test results.
        """
        if metric_fn is None:
            metric_fn = EvaluationMetrics.success_rate

        baseline_value = metric_fn(baseline_traces)
        optimized_value = metric_fn(optimized_traces)
        improvement = optimized_value - baseline_value

        # Permutation test for statistical significance
        combined = baseline_traces + optimized_traces
        n_baseline = len(baseline_traces)

        observed_diff = optimized_value - baseline_value

        # Permutation test
        n_permutations = 1000
        permuted_diffs = []

        for _ in range(n_permutations):
            # Shuffle and split
            shuffled = np.random.permutation(combined)
            perm_baseline = shuffled[:n_baseline]
            perm_optimized = shuffled[n_baseline:]

            perm_diff = metric_fn(list(perm_optimized)) - metric_fn(list(perm_baseline))
            permuted_diffs.append(perm_diff)

        # P-value: proportion of permutations with diff >= observed
        p_value = np.mean([abs(d) >= abs(observed_diff) for d in permuted_diffs])

        return {
            "baseline_value": baseline_value,
            "optimized_value": optimized_value,
            "improvement": improvement,
            "improvement_pct": (improvement / baseline_value * 100) if baseline_value > 0 else 0,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    @staticmethod
    def comprehensive_report(traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        return {
            "success_rate": EvaluationMetrics.success_rate(traces),
            "avg_tool_calls": EvaluationMetrics.avg_tool_calls(traces),
            "tool_call_efficiency": EvaluationMetrics.tool_call_efficiency(traces),
            "first_attempt_success": EvaluationMetrics.first_attempt_success_rate(traces),
            "error_recovery_rate": EvaluationMetrics.error_recovery_rate(traces),
            "avg_execution_time_ms": EvaluationMetrics.avg_execution_time(traces),
            "total_traces": len(traces),
        }
