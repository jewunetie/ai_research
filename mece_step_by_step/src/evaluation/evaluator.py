"""
MECE Evaluator - Orchestrates end-to-end evaluation of MECE vs baseline prompting.

This module ties together:
- Prompts (baseline and MECE variants)
- Model inference (Qwen3-0.6B-Instruct)
- Metrics (ME, CE, Accuracy)
- Results aggregation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from ..prompts.baseline_prompt import create_baseline_prompt_with_thinking
from ..prompts.mece_prompt import (
    create_mece_prompt_v1,
    create_mece_prompt_v2,
    create_mece_prompt_v3,
)
from ..metrics import (
    compute_accuracy,
    parse_reasoning_steps,
)

# Try to import ME/CE metrics (require dependencies)
try:
    from ..metrics import compute_mutual_exclusivity, compute_collective_exhaustiveness
    # Check if they're actually available (not None)
    HAS_FULL_METRICS = (
        compute_mutual_exclusivity is not None and
        compute_collective_exhaustiveness is not None
    )
except (ImportError, TypeError):
    HAS_FULL_METRICS = False
    compute_mutual_exclusivity = None
    compute_collective_exhaustiveness = None


class MECEEvaluator:
    """
    Orchestrates evaluation of MECE vs baseline prompting on math case analysis problems.

    Example usage:
        >>> evaluator = MECEEvaluator()
        >>> results = evaluator.evaluate_problem(problem, condition="baseline")
        >>> results['accuracy']['f1_score']
        0.95
    """

    def __init__(
        self,
        model_inference=None,
        mece_version: int = 1,
        verbose: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize MECE evaluator.

        Args:
            model_inference: Model inference object (e.g., QwenInference)
                           If None, will use mock responses for testing
            mece_version: Which MECE prompt version to use (1, 2, or 3)
            verbose: Print detailed evaluation information
            output_dir: Directory to save results (default: results/)
        """
        self.model_inference = model_inference
        self.mece_version = mece_version
        self.verbose = verbose
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Select MECE prompt function
        mece_funcs = {
            1: create_mece_prompt_v1,
            2: create_mece_prompt_v2,
            3: create_mece_prompt_v3,
        }
        self.mece_prompt_func = mece_funcs.get(mece_version, create_mece_prompt_v1)

        if not HAS_FULL_METRICS and verbose:
            print("⚠️  Full metrics (ME/CE) not available. Install dependencies with 'uv sync'")

    def evaluate_problem(
        self,
        problem: Dict[str, Any],
        condition: str = "baseline"
    ) -> Dict[str, Any]:
        """
        Evaluate a single problem with specified condition.

        Args:
            problem: Problem dictionary with 'problem', 'required_cases', 'ground_truth_solutions'
            condition: Either "baseline" or "mece"

        Returns:
            Dictionary containing:
            - problem_id: Problem ID
            - condition: Evaluation condition
            - prompt: Generated prompt
            - response: Model response
            - reasoning_steps: Extracted steps
            - metrics: Dict of ME, CE, and accuracy scores
            - latency_ms: Response generation time
            - timestamp: ISO timestamp

        Example:
            >>> problem = {"id": "abs_val_001", "problem": "Solve |x-3|=5", ...}
            >>> result = evaluator.evaluate_problem(problem, condition="baseline")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: {problem.get('id', 'unknown')}")
            print(f"Condition: {condition}")
            print(f"{'='*70}")

        # Generate prompt
        if condition == "baseline":
            prompt = create_baseline_prompt_with_thinking(problem)
        elif condition == "mece":
            prompt = self.mece_prompt_func(problem)
        else:
            raise ValueError(f"Unknown condition: {condition}. Use 'baseline' or 'mece'")

        if self.verbose:
            print(f"\nPrompt (first 200 chars):\n{prompt[:200]}...")

        # Generate response
        start_time = time.time()

        if self.model_inference is None:
            # Mock response for testing
            response = self._mock_response(problem, condition)
            if self.verbose:
                print("\n⚠️  Using mock response (no model loaded)")
        else:
            response = self.model_inference.generate(prompt)

        latency_ms = (time.time() - start_time) * 1000

        if self.verbose:
            print(f"\nResponse ({len(response)} chars, {latency_ms:.0f}ms):\n{response[:300]}...")

        # Parse reasoning steps
        reasoning_steps = parse_reasoning_steps(response)

        if self.verbose:
            print(f"\nExtracted {len(reasoning_steps)} reasoning steps")

        # Compute metrics
        metrics = self._compute_metrics(response, problem, reasoning_steps)

        if self.verbose:
            print(f"\nMetrics:")
            if 'accuracy' in metrics:
                print(f"  Accuracy F1: {metrics['accuracy']['f1_score']:.3f}")
            if 'mutual_exclusivity' in metrics:
                print(f"  ME Score: {metrics['mutual_exclusivity']['overall_me_score']:.3f}")
            if 'collective_exhaustiveness' in metrics:
                print(f"  CE Score: {metrics['collective_exhaustiveness']['overall_ce_score']:.3f}")

        # Package results
        result = {
            'problem_id': problem.get('id', 'unknown'),
            'category': problem.get('category', 'unknown'),
            'condition': condition,
            'mece_version': self.mece_version if condition == "mece" else None,
            'prompt': prompt,
            'response': response,
            'reasoning_steps': reasoning_steps,
            'n_steps': len(reasoning_steps),
            'metrics': metrics,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat(),
        }

        return result

    def evaluate_dataset(
        self,
        dataset_path: Path,
        condition: str = "baseline",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate entire dataset with specified condition.

        Args:
            dataset_path: Path to dataset JSON file
            condition: Either "baseline" or "mece"
            limit: Optional limit on number of problems to evaluate

        Returns:
            List of evaluation results
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            problems = json.load(f)

        if limit:
            problems = problems[:limit]

        print(f"\n{'='*70}")
        print(f"MECE EVALUATION - {condition.upper()} CONDITION")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"Problems: {len(problems)}")
        print(f"Condition: {condition}")
        if condition == "mece":
            print(f"MECE Version: {self.mece_version}")
        print(f"{'='*70}\n")

        # Evaluate each problem
        results = []
        for i, problem in enumerate(problems, 1):
            if self.verbose or True:  # Always show progress
                print(f"\nProblem {i}/{len(problems)}: {problem.get('id', 'unknown')}")

            result = self.evaluate_problem(problem, condition=condition)
            results.append(result)

        # Save results
        self._save_results(results, condition)

        return results

    def _compute_metrics(
        self,
        response: str,
        problem: Dict[str, Any],
        reasoning_steps: List[str]
    ) -> Dict[str, Any]:
        """Compute all metrics for a response."""
        metrics = {}

        # Accuracy (always available)
        ground_truth = problem.get('ground_truth_solutions', [])
        metrics['accuracy'] = compute_accuracy(
            response,
            ground_truth,
            normalize=True,
            verbose=False
        )

        # ME metrics (if dependencies available)
        if HAS_FULL_METRICS:
            try:
                metrics['mutual_exclusivity'] = compute_mutual_exclusivity(
                    response,
                    verbose=False
                )
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  ME computation failed: {e}")
                metrics['mutual_exclusivity'] = {'error': str(e)}

        # CE metrics (if dependencies available)
        if HAS_FULL_METRICS:
            try:
                metrics['collective_exhaustiveness'] = compute_collective_exhaustiveness(
                    response,
                    problem,
                    verbose=False
                )
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  CE computation failed: {e}")
                metrics['collective_exhaustiveness'] = {'error': str(e)}

        return metrics

    def _mock_response(self, problem: Dict[str, Any], condition: str) -> str:
        """Generate mock response for testing without model."""
        problem_text = problem.get('problem', 'Unknown problem')

        if condition == "baseline":
            return f"""Let me solve this problem step by step.

Problem: {problem_text}

Step 1: I'll analyze what cases need to be considered.
Step 2: For the first case, I'll solve the equation.
Step 3: For the second case, I'll solve the equation.
Step 4: I'll verify the solutions.

The solutions are x = 3 and x = -1.
"""
        else:  # mece
            return f"""Let me solve this using the MECE principle.

Problem: {problem_text}

**Case Identification:**
Case 1: When x >= 0
Case 2: When x < 0

**Case Analysis:**
Case 1 (x >= 0): Solving gives x = 3
Case 2 (x < 0): Solving gives x = -1

**Verification:**
All cases covered, no overlap.

**Final Answer:** x = 3 or x = -1
"""

    def _save_results(self, results: List[Dict[str, Any]], condition: str):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{condition}_results_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")
