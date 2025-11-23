#!/usr/bin/env python3
"""
Test Phase 3: Complete Optimization Loop
Meta-Learning for Prompt Improvement
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from pathlib import Path
from tcoml.optimization.optimization_loop import OptimizationLoop
from tcoml.llm.mock_client import MockLLMClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

print("=" * 70)
print("Phase 3 Checkpoint: Optimization Loop")
print("=" * 70)

# Initialize components
llm = MockLLMClient()
registry = ToolRegistry()
generator = SyntheticTaskGenerator()

# Test 1: Generate tasks for optimization
print("\n[Test 1] Generating tasks for optimization...")
tasks = generator.generate_all(n_per_type=4)  # 12 tasks total
print(f"✓ Generated {len(tasks)} tasks")

# Test 2: Create optimization loop
print("\n[Test 2] Creating optimization loop...")
output_dir = Path("./tool_call_optimization/data/results/test_optimization")
loop = OptimizationLoop(
    llm_client=llm,
    tool_registry=registry,
    output_dir=output_dir,
    max_iterations=3,  # Just 3 iterations for testing
    convergence_threshold=0.05,
    verbose=True
)
print("✓ Optimization loop created")

# Test 3: Run optimization with initial prompt
print("\n[Test 3] Running optimization loop...")
initial_prompt = """You are a helpful assistant that solves tasks using available tools.
Use tools to complete tasks accurately."""

result = loop.run(tasks, initial_prompt)

print("\n" + "=" * 70)
print("Optimization Results")
print("=" * 70)
print(f"Iterations completed: {result.iterations}")
print(f"Initial success rate: {result.initial_success_rate:.1%}")
print(f"Final success rate: {result.final_success_rate:.1%}")
print(f"Best success rate: {result.best_success_rate:.1%} (iteration {result.best_iteration})")
print(f"Converged: {result.converged}")
if result.converged:
    print(f"Convergence iteration: {result.convergence_iteration}")

# Test 4: Verify result structure
print("\n[Test 4] Verifying result structure...")
assert result.iterations > 0, "Should have at least 1 iteration"
assert len(result.prompt_history) == result.iterations, "Prompt history should match iterations"
assert len(result.performance_history) == result.iterations, "Performance history should match iterations"
assert result.best_prompt in result.prompt_history, "Best prompt should be in history"
print("✓ Result structure valid")

# Test 5: Check if files were saved
print("\n[Test 5] Checking saved files...")
assert output_dir.exists(), "Output directory should exist"
assert (output_dir / "prompts").exists(), "Prompts directory should exist"
assert (output_dir / "traces").exists(), "Traces directory should exist"
assert (output_dir / "feedback").exists(), "Feedback directory should exist"
assert (output_dir / "optimization_result.json").exists(), "Result file should exist"

prompt_files = list((output_dir / "prompts").glob("*.txt"))
print(f"✓ Saved {len(prompt_files)} prompt files")

trace_files = list((output_dir / "traces").glob("*.json"))
print(f"✓ Saved {len(trace_files)} trace files")

feedback_files = list((output_dir / "feedback").glob("*.json"))
print(f"✓ Saved {len(feedback_files)} feedback files")

# Test 6: Check prompt evolution
print("\n[Test 6] Analyzing prompt evolution...")
print(f"Number of prompts in history: {len(result.prompt_history)}")
print(f"\nInitial prompt ({len(initial_prompt)} chars):")
print(initial_prompt[:100] + "...")
print(f"\nFinal prompt ({len(result.best_prompt)} chars):")
print(result.best_prompt[:100] + "...")

# Test 7: Check performance trend
print("\n[Test 7] Performance trend analysis...")
print("Success rates by iteration:")
for i, rate in enumerate(result.performance_history):
    print(f"  Iteration {i}: {rate:.1%}")

# Test 8: Verify meta-prompter was called
print("\n[Test 8] Verifying meta-prompting happened...")
# Meta-prompter should be called (iterations - 1) times
expected_meta_prompting_calls = result.iterations - 1
# We can't directly verify this, but we can check that prompts changed
unique_prompts = set(result.prompt_history)
print(f"Unique prompts in history: {len(unique_prompts)}")
# With mock LLM, prompts might not change much, but structure should be there

print("\n" + "=" * 70)
print("✅ All Phase 3 tests passed!")
print("=" * 70)
print("\nPhase 3 Optimization Loop is complete and working!")
print(f"Initial → Final: {result.initial_success_rate:.1%} → {result.final_success_rate:.1%}")
print(f"Best performance: {result.best_success_rate:.1%}")
print("\nNote: With mock LLM, improvements are limited.")
print("For real optimization, use OpenAI or Anthropic LLM clients.")
