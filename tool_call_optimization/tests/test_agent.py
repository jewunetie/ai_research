#!/usr/bin/env python3
"""
Test the Tool Call Agent
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.agent.tool_call_agent import ToolCallAgent
from tcoml.llm.mock_client import MockLLMClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

print("=" * 70)
print("Tool Call Agent Test")
print("=" * 70)

# Initialize components
llm = MockLLMClient()
registry = ToolRegistry()
generator = SyntheticTaskGenerator()

# Create agent with a basic prompt
system_prompt = """You are a helpful assistant that solves tasks using available tools.
Use the tools provided to complete the task accurately.
Call tools as needed and use their results to answer the question."""

agent = ToolCallAgent(
    llm_client=llm,
    tool_registry=registry,
    system_prompt=system_prompt,
    verbose=True
)

print("\nSystem Prompt:")
print(system_prompt)
print("\n" + "=" * 70)

# Test 1: Simple calculator task
print("\n[Test 1] Solving calculator task...")
calc_tasks = generator.generate_calculator_tasks(5)
calc_task = calc_tasks[0]  # Should be a tip calculation
traces = agent.solve_tasks([calc_task])

assert len(traces) == 1
trace = traces[0]
print(f"\n✓ Task solved: {trace.task_success}")
print(f"  Tool calls made: {len(trace.tool_calls)}")
print(f"  Expected: {calc_task.ground_truth}")
print(f"  Got: {trace.final_output}")

# Test 2: String task
print("\n" + "=" * 70)
print("\n[Test 2] Solving string task...")
string_tasks = generator.generate_string_tasks(6)
string_task = string_tasks[0]  # Should be word count
traces = agent.solve_tasks([string_task])

assert len(traces) == 1
trace = traces[0]
print(f"\n✓ Task solved: {trace.task_success}")
print(f"  Tool calls made: {len(trace.tool_calls)}")
print(f"  Expected: {string_task.ground_truth}")
print(f"  Got: {trace.final_output}")

# Test 3: Batch solve multiple tasks
print("\n" + "=" * 70)
print("\n[Test 3] Solving batch of tasks...")
batch_tasks = generator.generate_all(n_per_type=2)
print(f"Solving {len(batch_tasks)} tasks...")

agent_quiet = ToolCallAgent(
    llm_client=llm,
    tool_registry=registry,
    system_prompt=system_prompt,
    verbose=False  # Quiet mode
)

batch_traces = agent_quiet.solve_tasks(batch_tasks)
print(f"\n✓ Solved {len(batch_traces)} tasks")

# Calculate success rate
success_count = sum(1 for t in batch_traces if t.task_success)
success_rate = success_count / len(batch_traces) if batch_traces else 0
print(f"  Success rate: {success_rate:.1%} ({success_count}/{len(batch_traces)})")

# Analyze failures
failed_traces = [t for t in batch_traces if not t.task_success]
if failed_traces:
    print(f"\n  Failed tasks:")
    for t in failed_traces[:3]:  # Show first 3 failures
        print(f"    - {t.task_id}: {t.task_description}")
        print(f"      Expected: {[task.ground_truth for task in batch_tasks if task.id == t.task_id][0]}")
        print(f"      Got: {t.final_output}")

# Test 4: Check trace quality
print("\n" + "=" * 70)
print("\n[Test 4] Checking trace quality...")
sample_trace = batch_traces[0]
print(f"Sample trace: {sample_trace.task_id}")
print(f"  Description: {sample_trace.task_description}")
print(f"  Tool calls: {len(sample_trace.tool_calls)}")
print(f"  Success: {sample_trace.task_success}")
print(f"  Duration: {sample_trace.total_time_ms:.2f}ms")

summary = sample_trace.get_tool_call_summary()
print(f"  Summary: {summary}")

print("\n" + "=" * 70)
print("✅ All agent tests passed!")
print("=" * 70)
print(f"\nAgent is working! Success rate: {success_rate:.1%}")
print("Ready to proceed with feedback generation.")
