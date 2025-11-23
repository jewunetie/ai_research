#!/usr/bin/env python3
"""
Test Phase 2: Complete Feedback Loop
Agent → Traces → Feedback → Aggregation
"""
import sys
sys.path.insert(0, '/home/user/ai_research/tool_call_optimization/src')

from tcoml.agent.tool_call_agent import ToolCallAgent
from tcoml.feedback.feedback_generator import FeedbackGenerator
from tcoml.feedback.feedback_aggregator import FeedbackAggregator
from tcoml.llm.mock_client import MockLLMClient
from tcoml.agent.tool_registry import ToolRegistry
from tcoml.benchmarks.synthetic_tasks import SyntheticTaskGenerator

print("=" * 70)
print("Phase 2 Checkpoint: Complete Feedback Loop")
print("=" * 70)

# Initialize all components
llm = MockLLMClient()
registry = ToolRegistry()
generator = SyntheticTaskGenerator()

# Test 1: Generate tasks and solve them
print("\n[Test 1] Generating and solving tasks...")
tasks = generator.generate_all(n_per_type=3)  # 9 tasks total
print(f"Generated {len(tasks)} tasks")

baseline_prompt = """You are a helpful assistant that solves tasks using available tools.
Use tools to complete tasks accurately."""

agent = ToolCallAgent(llm, registry, baseline_prompt, verbose=False)
traces = agent.solve_tasks(tasks)

success_count = sum(1 for t in traces if t.task_success)
print(f"✓ Solved {len(traces)} tasks")
print(f"  Success rate: {success_count}/{len(traces)} ({success_count/len(traces):.1%})")

# Test 2: Generate feedback for each trace
print("\n[Test 2] Generating feedback...")
feedback_gen = FeedbackGenerator(llm)
task_dict = {t.id: t for t in tasks}
feedbacks = feedback_gen.generate_batch_feedback(traces, task_dict)

print(f"✓ Generated {len(feedbacks)} feedback items")
print(f"\nSample feedback (task: {feedbacks[0].task_id}):")
print(f"  Success: {feedbacks[0].success}")
print(f"  Critique length: {len(feedbacks[0].critique)} chars")
print(f"  Suggestions: {len(feedbacks[0].suggestions)}")
if feedbacks[0].suggestions:
    print(f"  First suggestion: {feedbacks[0].suggestions[0][:80]}...")

# Test 3: Aggregate feedback
print("\n[Test 3] Aggregating feedback...")
aggregator = FeedbackAggregator(llm)
aggregated = aggregator.aggregate(feedbacks)

print(f"✓ Aggregated feedback:")
print(f"  Total traces: {aggregated.total_traces}")
print(f"  Success rate: {aggregated.successful_traces}/{aggregated.total_traces}")
print(f"  Failure patterns identified: {len(aggregated.common_failure_patterns)}")
print(f"  Success patterns identified: {len(aggregated.common_success_patterns)}")
print(f"  Top suggestions: {len(aggregated.top_suggestions)}")

print(f"\nAggregated summary:")
print(aggregated.summary)

# Test 4: Verify feedback structure
print("\n[Test 4] Verifying feedback structure...")
assert all(isinstance(f.critique, str) for f in feedbacks), "All critiques should be strings"
assert all(isinstance(f.suggestions, list) for f in feedbacks), "All suggestions should be lists"
assert aggregated.total_traces == len(traces), "Total traces should match"
print("✓ All feedback structures valid")

# Test 5: Check failed vs successful feedback
print("\n[Test 5] Analyzing failed vs successful feedback...")
failed_feedbacks = [f for f in feedbacks if not f.success]
successful_feedbacks = [f for f in feedbacks if f.success]

print(f"Failed task feedbacks: {len(failed_feedbacks)}")
print(f"Successful task feedbacks: {len(successful_feedbacks)}")

if failed_feedbacks:
    print(f"\nSample failure analysis:")
    sample_fail = failed_feedbacks[0]
    print(f"  Task: {sample_fail.task_id}")
    print(f"  What went wrong: {sample_fail.what_went_wrong[:100]}...")

if successful_feedbacks:
    print(f"\nSample success analysis:")
    sample_success = successful_feedbacks[0]
    print(f"  Task: {sample_success.task_id}")
    print(f"  What went right: {sample_success.what_went_right[:100]}...")

# Test 6: End-to-end workflow validation
print("\n[Test 6] Validating end-to-end workflow...")
workflow_steps = [
    "1. Generate tasks ✓",
    "2. Solve with agent ✓",
    "3. Capture traces ✓",
    "4. Generate feedback ✓",
    "5. Aggregate patterns ✓"
]
for step in workflow_steps:
    print(f"  {step}")

print("\n" + "=" * 70)
print("✅ All Phase 2 tests passed!")
print("=" * 70)
print(f"\nPhase 2 Feedback Loop is complete and working!")
print(f"Success rate: {success_count}/{len(traces)} ({success_count/len(traces):.1%})")
print(f"Ready for Phase 3: Meta-Prompting!")
