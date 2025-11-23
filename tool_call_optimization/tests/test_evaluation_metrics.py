#!/usr/bin/env python3
"""
Test Evaluation Metrics
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.evaluation.metrics import EvaluationMetrics
from tcoml.tracing.trace_models import ExecutionTrace, ToolCall
from datetime import datetime

print("=" * 70)
print("Evaluation Metrics Test")
print("=" * 70)

# Create sample traces for testing
def create_trace(task_id: str, success: bool, tool_calls: list) -> ExecutionTrace:
    """Helper to create test traces"""
    trace = ExecutionTrace(
        task_id=task_id,
        task_description=f"Task {task_id}",
        system_prompt="Test prompt",
        task_success=success,
        total_time_ms=100.0
    )
    for tc in tool_calls:
        trace.add_tool_call(tc)
    return trace

# Test 1: Success rate
print("\n[Test 1] Testing success rate...")
traces = [
    create_trace("1", True, []),
    create_trace("2", False, []),
    create_trace("3", True, []),
]
success_rate = EvaluationMetrics.success_rate(traces)
print(f"✓ Success rate: {success_rate:.1%}")
assert success_rate == 2/3, f"Expected 66.7%, got {success_rate:.1%}"

# Test 2: Average tool calls
print("\n[Test 2] Testing average tool calls...")
traces = [
    create_trace("1", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
    create_trace("2", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
]
avg_calls = EvaluationMetrics.avg_tool_calls(traces)
print(f"✓ Average tool calls: {avg_calls:.2f}")
assert avg_calls == 1.5, f"Expected 1.5, got {avg_calls}"

# Test 3: Tool call efficiency
print("\n[Test 3] Testing tool call efficiency...")
traces = [
    create_trace("1", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
        ToolCall(tool_name="calc", arguments={}, success=False, duration_ms=1.0),
    ]),
    create_trace("2", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
]
efficiency = EvaluationMetrics.tool_call_efficiency(traces)
print(f"✓ Tool call efficiency: {efficiency:.1%}")
assert efficiency == 0.75, f"Expected 75%, got {efficiency:.1%}"

# Test 4: First attempt success rate
print("\n[Test 4] Testing first attempt success rate...")
traces = [
    create_trace("1", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
    create_trace("2", True, [
        ToolCall(tool_name="calc", arguments={}, success=False, duration_ms=1.0),
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
    create_trace("3", False, []),
]
first_attempt = EvaluationMetrics.first_attempt_success_rate(traces)
print(f"✓ First attempt success: {first_attempt:.1%}")
assert first_attempt == 1/3, f"Expected 33.3%, got {first_attempt:.1%}"

# Test 5: Error recovery rate
print("\n[Test 5] Testing error recovery rate...")
traces = [
    create_trace("1", True, [
        ToolCall(tool_name="calc", arguments={}, success=False, duration_ms=1.0),
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),  # Recovered from error
    create_trace("2", False, [
        ToolCall(tool_name="calc", arguments={}, success=False, duration_ms=1.0),
    ]),  # Had error, didn't recover
]
recovery_rate = EvaluationMetrics.error_recovery_rate(traces)
print(f"✓ Error recovery rate: {recovery_rate:.1%}")
assert recovery_rate == 0.5, f"Expected 50%, got {recovery_rate:.1%}"

# Test 6: Comprehensive report
print("\n[Test 6] Testing comprehensive report...")
traces = [
    create_trace("1", True, [
        ToolCall(tool_name="calc", arguments={}, success=True, duration_ms=1.0),
    ]),
    create_trace("2", False, [
        ToolCall(tool_name="calc", arguments={}, success=False, duration_ms=1.0),
    ]),
]
report = EvaluationMetrics.comprehensive_report(traces)
print("✓ Comprehensive report generated:")
for key, value in report.items():
    print(f"  {key}: {value}")

# Verify report structure
assert "success_rate" in report
assert "avg_tool_calls" in report
assert "tool_call_efficiency" in report
assert "first_attempt_success" in report
assert "error_recovery_rate" in report
assert "avg_execution_time_ms" in report
assert "total_traces" in report
print("✓ Report structure valid")

# Test 7: Compare performance
print("\n[Test 7] Testing performance comparison...")
baseline = [
    create_trace("1", False, []),
    create_trace("2", True, []),
]
optimized = [
    create_trace("1", True, []),
    create_trace("2", True, []),
]
comparison = EvaluationMetrics.compare_performance(baseline, optimized)
print(f"✓ Performance comparison:")
print(f"  Baseline: {comparison['baseline_value']:.1%}")
print(f"  Optimized: {comparison['optimized_value']:.1%}")
print(f"  Improvement: {comparison['improvement']:.1%}")
print(f"  P-value: {comparison['p_value']:.3f}")
print(f"  Significant: {comparison['significant']}")

# Test 8: Empty traces handling
print("\n[Test 8] Testing empty traces handling...")
empty_traces = []
rate = EvaluationMetrics.success_rate(empty_traces)
assert rate == 0.0, "Empty traces should return 0.0"
avg = EvaluationMetrics.avg_tool_calls(empty_traces)
assert avg == 0.0, "Empty traces should return 0.0"
print("✓ Empty traces handled correctly")

print("\n" + "=" * 70)
print("✅ All evaluation metrics tests passed!")
print("=" * 70)
print("\nPhase 4 Evaluation Metrics are working correctly!")
