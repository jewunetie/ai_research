#!/usr/bin/env python3
"""
Checkpoint test for Phase 1A: Tasks and Tools
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.benchmarks.task_models import Task, TaskDifficulty
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator
from tcoml.agent.tool_registry import ToolRegistry

print("=" * 70)
print("Phase 1A Checkpoint Test: Tasks and Tools")
print("=" * 70)

# Test 1: Generate tasks
print("\n[Test 1] Generating tasks...")
generator = SyntheticTaskGenerator()
tasks = generator.generate_calculator_tasks(5)
print(f"✓ Generated {len(tasks)} tasks")
print(f"  Example task: {tasks[0].id} - {tasks[0].description}")
print(f"  Ground truth: {tasks[0].ground_truth}")

# Test 2: Tool registry
print("\n[Test 2] Testing tool registry...")
registry = ToolRegistry()
available_tools = registry.list_tools()
print(f"✓ Available tools: {available_tools}")

# Test 3: Execute calculator tool
print("\n[Test 3] Testing calculator tool...")
result = registry.get("calculator").execute(operation="add", a=10, b=20)
print(f"✓ Calculator add(10, 20): {result}")
assert result["success"] == True
assert result["result"] == 30
print(f"  Success: {result['success']}, Result: {result['result']}")

# Test 4: Execute percentage calculation
print("\n[Test 4] Testing percentage calculation...")
result = registry.get("calculator").execute(operation="percentage", a=100, b=15)
print(f"✓ Calculator percentage(100, 15%): {result}")
assert result["success"] == True
assert result["result"] == 15.0
print(f"  Success: {result['success']}, Result: {result['result']}")

# Test 5: Test string processor
print("\n[Test 5] Testing string processor...")
result = registry.get("string_processor").execute(operation="reverse", text="hello")
print(f"✓ String reverse('hello'): {result}")
assert result["success"] == True
assert result["result"] == "olleh"
print(f"  Success: {result['success']}, Result: {result['result']}")

# Test 6: Test task evaluation
print("\n[Test 6] Testing task evaluation...")
task = tasks[0]
# Simulate correct answer
correct_output = task.ground_truth
is_correct = task.evaluate(correct_output)
print(f"✓ Task evaluation with correct output: {is_correct}")
assert is_correct == True

# Test with wrong answer
wrong_output = task.ground_truth + 100 if task.ground_truth else "wrong"
is_wrong = task.evaluate(wrong_output)
print(f"✓ Task evaluation with wrong output: {is_wrong}")
assert is_wrong == False

# Test 7: Test error handling
print("\n[Test 7] Testing error handling...")
result = registry.get("calculator").execute(operation="divide", a=10, b=0)
print(f"✓ Division by zero handling: {result}")
assert result["success"] == False
assert "error" in result
print(f"  Success: {result['success']}, Error: {result['error']}")

# Test 8: Generate all task types
print("\n[Test 8] Generating all task types...")
all_tasks = generator.generate_all(n_per_type=3)
print(f"✓ Generated {len(all_tasks)} tasks total")
print(f"  Calculator tasks: {sum(1 for t in all_tasks if t.category == 'calculator')}")
print(f"  String tasks: {sum(1 for t in all_tasks if t.category == 'string')}")
print(f"  Multi-step tasks: {sum(1 for t in all_tasks if t.category == 'multi_step')}")

print("\n" + "=" * 70)
print("✅ All Phase 1A tests passed!")
print("=" * 70)
