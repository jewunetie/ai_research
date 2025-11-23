#!/usr/bin/env python3
"""
Checkpoint test for Phase 1B: Tracing
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.tracing.execution_tracer import ExecutionTracer
from tcoml.agent.tool_registry import ToolRegistry

print("=" * 70)
print("Phase 1B Checkpoint Test: Execution Tracing")
print("=" * 70)

# Initialize components
registry = ToolRegistry()
tracer = ExecutionTracer(registry)

# Test 1: Start a trace
print("\n[Test 1] Starting execution trace...")
tracer.start_trace("test_001", "Calculate 10+20", "You are a helpful assistant")
print("✓ Trace started")

# Test 2: Record successful tool call
print("\n[Test 2] Recording tool calls...")
result = tracer.record_tool_call("calculator", {"operation": "add", "a": 10, "b": 20})
print(f"✓ Tool call result: {result}")
assert result["success"] == True
assert result["result"] == 30

# Test 3: Record another tool call
result2 = tracer.record_tool_call("calculator", {"operation": "multiply", "a": 5, "b": 6})
print(f"✓ Second tool call result: {result2}")
assert result2["success"] == True
assert result2["result"] == 30

# Test 4: End trace
print("\n[Test 3] Ending trace...")
trace = tracer.end_trace(final_output=result["result"], task_success=True)
print(f"✓ Trace ended. Task success: {trace.task_success}")
assert trace.task_success == True
assert len(trace.tool_calls) == 2

# Test 5: Get trace summary
print("\n[Test 4] Generating trace summary...")
summary = tracer.get_trace_summary(trace)
print(f"✓ Trace summary generated:\n{summary}")

# Test 6: Test trace with error
print("\n[Test 5] Testing trace with error...")
tracer.start_trace("test_002", "Divide by zero test", "You are a helpful assistant")
error_result = tracer.record_tool_call("calculator", {"operation": "divide", "a": 10, "b": 0})
print(f"✓ Error handled: {error_result}")
assert error_result["success"] == False

error_trace = tracer.end_trace(final_output=None, task_success=False, error="Division by zero")
print(f"✓ Error trace ended. Success: {error_trace.task_success}")
assert error_trace.task_success == False

# Test 7: Check trace statistics
print("\n[Test 6] Checking trace statistics...")
stats = trace.get_tool_call_summary()
print(f"✓ Trace statistics: {stats}")
assert stats["total_calls"] == 2
assert stats["successful_calls"] == 2
assert stats["failed_calls"] == 0

error_stats = error_trace.get_tool_call_summary()
print(f"✓ Error trace statistics: {error_stats}")
assert error_stats["total_calls"] == 1
assert error_stats["successful_calls"] == 0
assert error_stats["failed_calls"] == 1

print("\n" + "=" * 70)
print("✅ All Phase 1B tests passed!")
print("=" * 70)
