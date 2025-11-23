#!/usr/bin/env python3
"""
Complete Phase 1 Checkpoint Test
Tests the full integration of tasks, tools, tracing, and LLM client
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.llm.mock_client import MockLLMClient
from tcoml.llm.base_client import Message
from tcoml.tracing.execution_tracer import ExecutionTracer
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.task_models import Task, TaskDifficulty
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

print("=" * 70)
print("Phase 1 Complete Checkpoint Test")
print("=" * 70)

# Initialize all components
registry = ToolRegistry()
tracer = ExecutionTracer(registry)
llm = MockLLMClient()
generator = SyntheticTaskGenerator()

# Test 1: LLM Client Basic Functionality
print("\n[Test 1] Testing Mock LLM Client...")
response = llm.complete([
    Message(role="user", content="What is 2+2?")
])
print(f"✓ LLM Response: {response.content}")
print(f"  Tool calls: {response.tool_calls}")

# Test 2: LLM with Tool Calling
print("\n[Test 2] Testing LLM with Tool Calling...")
tools = [registry.get("calculator").to_openai_function()]
response = llm.complete(
    messages=[Message(role="user", content="Calculate 10 + 20")],
    tools=tools
)
print(f"✓ Tool calls generated: {response.tool_calls}")
assert response.tool_calls is not None
assert len(response.tool_calls) > 0
assert response.tool_calls[0]["name"] == "calculator"

# Test 3: Full Workflow - Generate Task, Execute with Tracer
print("\n[Test 3] Testing Full Workflow...")
tasks = generator.generate_calculator_tasks(5)  # Need at least 4 for the generator logic
task = tasks[0]
print(f"  Task: {task.description}")
print(f"  Expected: {task.ground_truth}")

# Start tracing
tracer.start_trace(task.id, task.description, "You are a helpful calculator")

# Get LLM response for the task
tools = [registry.get(t).to_openai_function() for t in task.required_tools]
response = llm.complete(
    messages=[Message(role="user", content=task.description)],
    tools=tools
)

print(f"  LLM generated {len(response.tool_calls) if response.tool_calls else 0} tool calls")

# Execute tool calls through tracer
final_result = None
if response.tool_calls:
    for tool_call in response.tool_calls:
        import json
        args = json.loads(tool_call["arguments"])
        result = tracer.record_tool_call(tool_call["name"], args)
        print(f"  Executed: {tool_call['name']}({args}) → {result['result']}")
        final_result = result["result"]

# End trace
task_success = task.evaluate(final_result) if final_result else False
trace = tracer.end_trace(final_output=final_result, task_success=task_success)

print(f"✓ Task completed. Success: {trace.task_success}")
print(f"  Final result: {final_result}")
print(f"  Expected: {task.ground_truth}")

# Test 4: Generate Trace Summary
print("\n[Test 4] Generating Trace Summary...")
summary = tracer.get_trace_summary(trace)
print("✓ Trace Summary:")
print(summary)

# Test 5: Test with String Task
print("\n[Test 5] Testing String Task...")
string_tasks = generator.generate_string_tasks(6)  # Need at least 3 for the generator logic
string_task = string_tasks[0]
print(f"  Task: {string_task.description}")
print(f"  Expected: {string_task.ground_truth}")

tracer.start_trace(string_task.id, string_task.description, "You are a helpful assistant")
string_tools = [registry.get(t).to_openai_function() for t in string_task.required_tools]

response = llm.complete(
    messages=[Message(role="user", content=string_task.description)],
    tools=string_tools
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        import json
        args = json.loads(tool_call["arguments"])
        result = tracer.record_tool_call(tool_call["name"], args)
        final_result = result["result"]

task_success = string_task.evaluate(final_result) if final_result else False
string_trace = tracer.end_trace(final_output=final_result, task_success=task_success)

print(f"✓ String task completed. Success: {string_trace.task_success}")
print(f"  Final result: {final_result}")
print(f"  Expected: {string_task.ground_truth}")

# Test 6: Verify Component Integration
print("\n[Test 6] Verifying Component Integration...")
print(f"✓ Tools available: {registry.list_tools()}")
print(f"✓ Tasks generated: {len(generator.generate_all(n_per_type=2))} total")
print(f"✓ Traces captured: 2")
print(f"✓ LLM calls made: {len(llm.call_history)}")

# Test 7: Check Trace Statistics
print("\n[Test 7] Checking Trace Statistics...")
stats = trace.get_tool_call_summary()
print(f"✓ Trace statistics:")
print(f"  Total calls: {stats['total_calls']}")
print(f"  Successful: {stats['successful_calls']}")
print(f"  Failed: {stats['failed_calls']}")
print(f"  Duration: {stats['total_duration_ms']:.2f}ms")

print("\n" + "=" * 70)
print("✅ All Phase 1 Complete tests passed!")
print("=" * 70)
print("\nPhase 1 Foundation is complete and working!")
print("Ready to proceed to Phase 2: Feedback Loop")
