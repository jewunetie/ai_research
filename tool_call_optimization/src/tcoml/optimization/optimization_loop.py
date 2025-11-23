import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from ..agent.tool_call_agent import ToolCallAgent
from ..agent.tool_registry import ToolRegistry
from ..benchmarks.task_models import Task
from ..feedback.feedback_generator import FeedbackGenerator
from ..feedback.feedback_aggregator import FeedbackAggregator, AggregatedFeedback
from ..llm.base_client import BaseLLMClient
from ..tracing.trace_models import ExecutionTrace, TraceDataset
from .meta_prompter import MetaPrompter, PromptUpdate


class OptimizationResult(BaseModel):
    """Results from optimization loop"""
    iterations: int
    initial_success_rate: float
    final_success_rate: float
    best_success_rate: float
    best_iteration: int
    best_prompt: str
    prompt_history: List[str]
    performance_history: List[float]
    converged: bool
    convergence_iteration: Optional[int] = None


class OptimizationLoop:
    """
    Main optimization loop orchestrating the meta-learning process.

    Implements the full Arize-ai inspired workflow:
    1. Train/Test Split
    2. Execute tasks with current prompt
    3. Generate feedback
    4. Aggregate feedback
    5. Meta-prompt to improve
    6. Re-evaluate
    7. Repeat
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        output_dir: Path,
        max_iterations: int = 10,
        convergence_threshold: float = 0.02,
        train_test_split: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize optimization loop.

        Args:
            llm_client: LLM client for agent and meta-prompting
            tool_registry: Available tools
            output_dir: Directory to save results
            max_iterations: Maximum optimization iterations
            convergence_threshold: Success rate change threshold for convergence
            train_test_split: Fraction of tasks for training (rest for testing)
            verbose: Print progress
        """
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.train_test_split = train_test_split
        self.verbose = verbose

        # Initialize components
        self.feedback_generator = FeedbackGenerator(llm_client)
        self.feedback_aggregator = FeedbackAggregator(llm_client)
        self.meta_prompter = MetaPrompter(llm_client, verbose=verbose)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir(exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "feedback").mkdir(exist_ok=True)

    def run(
        self,
        tasks: List[Task],
        initial_prompt: str
    ) -> OptimizationResult:
        """
        Run the full optimization loop.

        Args:
            tasks: List of tasks for training and testing
            initial_prompt: Starting system prompt

        Returns:
            OptimizationResult with performance and prompts
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Optimization Loop")
            print(f"{'='*70}")
            print(f"Total tasks: {len(tasks)}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Convergence threshold: {self.convergence_threshold}")

        # Step 1: Train/Test Split
        train_tasks, test_tasks = self._split_tasks(tasks)

        if self.verbose:
            print(f"\nTrain tasks: {len(train_tasks)}")
            print(f"Test tasks: {len(test_tasks)}")

        # Initialize tracking
        current_prompt = initial_prompt
        prompt_history = [initial_prompt]
        performance_history = []
        best_success_rate = 0.0
        best_iteration = 0
        best_prompt = initial_prompt
        converged = False
        convergence_iteration = None

        # Save initial prompt
        self._save_prompt(initial_prompt, 0)

        # Main optimization loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*70}")

            # Step 2: Execute tasks with current prompt
            agent = ToolCallAgent(
                llm_client=self.llm,
                tool_registry=self.tool_registry,
                system_prompt=current_prompt,
                verbose=False
            )

            traces = agent.solve_tasks(train_tasks)

            # Calculate success rate
            success_count = sum(1 for t in traces if t.task_success)
            success_rate = success_count / len(train_tasks)
            performance_history.append(success_rate)

            if self.verbose:
                print(f"\nTrain Success Rate: {success_rate:.1%} ({success_count}/{len(train_tasks)})")

            # Track best
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_iteration = iteration
                best_prompt = current_prompt

            # Save traces
            self._save_traces(traces, iteration)

            # Step 3 & 4: Generate and aggregate feedback
            task_dict = {t.id: t for t in train_tasks}
            feedbacks = self.feedback_generator.generate_batch_feedback(traces, task_dict)
            aggregated_feedback = self.feedback_aggregator.aggregate(feedbacks)

            # Save feedback
            self._save_feedback(aggregated_feedback, iteration)

            # Check convergence (if not first iteration)
            if iteration > 0:
                improvement = success_rate - performance_history[-2]
                if abs(improvement) < self.convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    if self.verbose:
                        print(f"\n✓ Converged! Improvement {improvement:.1%} < threshold {self.convergence_threshold:.1%}")
                    break

            # Step 5: Meta-prompt to generate improved prompt
            if iteration < self.max_iterations - 1:  # Don't update on last iteration
                prompt_update = self.meta_prompter.improve_prompt(
                    current_prompt=current_prompt,
                    feedback=aggregated_feedback,
                    iteration=iteration + 1,
                    previous_prompts=prompt_history
                )

                current_prompt = prompt_update.new_prompt
                prompt_history.append(current_prompt)

                # Save new prompt
                self._save_prompt(current_prompt, iteration + 1)

                if self.verbose:
                    print(f"\nPrompt updated for next iteration")

        # Step 6: Final evaluation on test set
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Final Evaluation on Test Set")
            print(f"{'='*70}")

        test_agent = ToolCallAgent(
            llm_client=self.llm,
            tool_registry=self.tool_registry,
            system_prompt=best_prompt,
            verbose=False
        )

        test_traces = test_agent.solve_tasks(test_tasks)
        test_success_count = sum(1 for t in test_traces if t.task_success)
        final_success_rate = test_success_count / len(test_tasks) if test_tasks else 0

        if self.verbose:
            print(f"\nTest Success Rate: {final_success_rate:.1%} ({test_success_count}/{len(test_tasks)})")
            print(f"Best Train Success Rate: {best_success_rate:.1%} (Iteration {best_iteration})")

        # Save test traces
        self._save_traces(test_traces, iteration, prefix="test")

        # Create result
        result = OptimizationResult(
            iterations=iteration + 1,
            initial_success_rate=performance_history[0],
            final_success_rate=final_success_rate,
            best_success_rate=best_success_rate,
            best_iteration=best_iteration,
            best_prompt=best_prompt,
            prompt_history=prompt_history,
            performance_history=performance_history,
            converged=converged,
            convergence_iteration=convergence_iteration
        )

        # Save final result
        self._save_result(result)

        return result

    def _split_tasks(self, tasks: List[Task]) -> tuple:
        """Split tasks into train and test sets"""
        # Shuffle with fixed seed for reproducibility
        shuffled = tasks.copy()
        random.seed(42)
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * self.train_test_split)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        return train, test

    def _save_prompt(self, prompt: str, iteration: int) -> None:
        """Save prompt to file"""
        filename = self.output_dir / "prompts" / f"prompt_iter_{iteration:03d}.txt"
        filename.write_text(prompt)

    def _save_traces(self, traces: List[ExecutionTrace], iteration: int, prefix: str = "train") -> None:
        """Save traces to JSON"""
        dataset = TraceDataset(traces=traces)
        filename = self.output_dir / "traces" / f"{prefix}_traces_iter_{iteration:03d}.json"
        filename.write_text(dataset.model_dump_json(indent=2))

    def _save_feedback(self, feedback: AggregatedFeedback, iteration: int) -> None:
        """Save aggregated feedback"""
        filename = self.output_dir / "feedback" / f"feedback_iter_{iteration:03d}.json"
        filename.write_text(feedback.model_dump_json(indent=2))

    def _save_result(self, result: OptimizationResult) -> None:
        """Save final optimization result"""
        filename = self.output_dir / "optimization_result.json"
        filename.write_text(result.model_dump_json(indent=2))

        if self.verbose:
            print(f"\n✓ Results saved to {self.output_dir}")
