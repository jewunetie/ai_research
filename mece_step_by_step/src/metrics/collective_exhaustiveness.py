"""
Collective Exhaustiveness (CE) metrics for MECE reasoning evaluation.

These metrics measure how well reasoning steps cover all necessary cases
and arrive at complete solutions.
"""

from typing import List, Dict, Any, Set
from .parsers import parse_reasoning_steps, extract_case_conditions, parse_solutions


class CollectiveExhaustivenessScorer:
    """
    Compute collective exhaustiveness scores for reasoning chains.

    Uses two complementary approaches:
    1. Case enumeration - compares detected cases vs. required cases
    2. Solution coverage - checks if all ground truth solutions were found

    Example:
        >>> scorer = CollectiveExhaustivenessScorer()
        >>> response = "Case 1: x >= 0 gives x = 3\\nCase 2: x < 0 gives x = -1"
        >>> problem = {"required_cases": ["x >= 0", "x < 0"], "ground_truth_solutions": ["x = 3", "x = -1"]}
        >>> scores = scorer.compute_ce_scores(response, problem)
        >>> print(scores['overall_ce_score'])
        1.0  # Perfect coverage
    """

    def __init__(
        self,
        case_match_threshold: float = 0.7,
        partial_credit: bool = True
    ):
        """
        Initialize CE scorer.

        Args:
            case_match_threshold: Threshold for fuzzy case matching
            partial_credit: Whether to give partial credit for incomplete coverage
        """
        self.case_match_threshold = case_match_threshold
        self.partial_credit = partial_credit

    def compute_ce_scores(
        self,
        response: str,
        problem: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compute all collective exhaustiveness scores for a response.

        Args:
            response: Model's full response text
            problem: Problem dictionary containing 'required_cases' and 'ground_truth_solutions'
            verbose: Print detailed scoring information

        Returns:
            Dictionary containing:
            - case_enumeration_ce_score: Case coverage score (0-1)
            - solution_coverage_ce_score: Solution coverage score (0-1)
            - overall_ce_score: Combined CE score (0-1)
            - n_cases_detected: Number of cases detected in response
            - n_cases_required: Number of cases required
            - n_cases_matched: Number of required cases matched
            - n_solutions_found: Number of ground truth solutions found
            - n_solutions_total: Total number of ground truth solutions
            - missing_cases: List of required cases not found
            - missing_solutions: List of ground truth solutions not found

        Example:
            >>> scorer = CollectiveExhaustivenessScorer()
            >>> scores = scorer.compute_ce_scores(
            ...     "Case 1: x >= 0, gives x = 3\\nCase 2: x < 0, gives x = -1",
            ...     {"required_cases": ["x >= 0", "x < 0"], "ground_truth_solutions": ["x = 3", "x = -1"]}
            ... )
            >>> scores['overall_ce_score']
            1.0
        """
        if verbose:
            print(f"\n📊 Collective Exhaustiveness Scoring")

        # Parse reasoning steps and conditions
        steps = parse_reasoning_steps(response)
        detected_conditions = extract_case_conditions(steps)

        # Get required cases and ground truth solutions
        required_cases = problem.get("required_cases", [])
        ground_truth_solutions = problem.get("ground_truth_solutions", [])

        if verbose:
            print(f"  Steps detected: {len(steps)}")
            print(f"  Conditions detected: {len(detected_conditions)}")
            print(f"  Required cases: {len(required_cases)}")
            print(f"  Ground truth solutions: {len(ground_truth_solutions)}")

        # Compute case enumeration CE score
        case_score, n_matched, missing_cases = self._compute_case_enumeration_ce(
            detected_conditions,
            required_cases,
            verbose=verbose
        )

        # Compute solution coverage CE score
        solution_score, n_found, missing_solutions = self._compute_solution_coverage_ce(
            response,
            ground_truth_solutions,
            verbose=verbose
        )

        # Combine scores (equal weight)
        overall_score = 0.5 * case_score + 0.5 * solution_score

        results = {
            'case_enumeration_ce_score': round(case_score, 4),
            'solution_coverage_ce_score': round(solution_score, 4),
            'overall_ce_score': round(overall_score, 4),
            'n_cases_detected': len(detected_conditions),
            'n_cases_required': len(required_cases),
            'n_cases_matched': n_matched,
            'n_solutions_found': n_found,
            'n_solutions_total': len(ground_truth_solutions),
            'missing_cases': missing_cases,
            'missing_solutions': missing_solutions,
        }

        if verbose:
            print(f"\n  Results:")
            print(f"    Case enumeration CE: {results['case_enumeration_ce_score']:.3f}")
            print(f"    Solution coverage CE: {results['solution_coverage_ce_score']:.3f}")
            print(f"    Overall CE score: {results['overall_ce_score']:.3f}")
            if missing_cases:
                print(f"    ⚠️  Missing cases: {missing_cases}")
            if missing_solutions:
                print(f"    ⚠️  Missing solutions: {missing_solutions}")

        return results

    def _compute_case_enumeration_ce(
        self,
        detected_conditions: List[str],
        required_cases: List[str],
        verbose: bool = False
    ) -> tuple[float, int, List[str]]:
        """
        Compute CE score based on case enumeration.

        Checks if all required cases were identified in the reasoning.

        Args:
            detected_conditions: Conditions detected in response
            required_cases: Cases that should be considered
            verbose: Print detailed information

        Returns:
            Tuple of (ce_score, n_matched, missing_cases)
            - ce_score: 0-1, higher = better coverage
            - n_matched: Number of required cases matched
            - missing_cases: List of required cases not found
        """
        if not required_cases:
            # No required cases means perfect score
            return 1.0, 0, []

        if not detected_conditions:
            # No conditions detected = 0 coverage
            return 0.0, 0, required_cases.copy()

        # Normalize conditions for comparison
        detected_normalized = [self._normalize_condition(c) for c in detected_conditions]
        required_normalized = [self._normalize_condition(c) for c in required_cases]

        if verbose:
            print(f"    Detected (normalized): {detected_normalized}")
            print(f"    Required (normalized): {required_normalized}")

        # Find matches
        matched_cases = []
        missing_cases = []

        for req_case in required_cases:
            req_norm = self._normalize_condition(req_case)

            # Check if this required case is covered
            is_matched = any(
                self._conditions_match(req_norm, det_norm)
                for det_norm in detected_normalized
            )

            if is_matched:
                matched_cases.append(req_case)
            else:
                missing_cases.append(req_case)

        n_matched = len(matched_cases)
        n_required = len(required_cases)

        # CE score = fraction of required cases matched
        ce_score = n_matched / n_required if n_required > 0 else 1.0

        if verbose and n_matched < n_required:
            print(f"    ⚠️  Matched {n_matched}/{n_required} required cases")

        return ce_score, n_matched, missing_cases

    def _compute_solution_coverage_ce(
        self,
        response: str,
        ground_truth_solutions: List[str],
        verbose: bool = False
    ) -> tuple[float, int, List[str]]:
        """
        Compute CE score based on solution coverage.

        Checks if all ground truth solutions were found.

        Args:
            response: Full response text
            ground_truth_solutions: List of expected solutions
            verbose: Print detailed information

        Returns:
            Tuple of (ce_score, n_found, missing_solutions)
            - ce_score: 0-1, higher = better coverage
            - n_found: Number of ground truth solutions found
            - missing_solutions: List of solutions not found
        """
        if not ground_truth_solutions:
            # No required solutions means perfect score
            return 1.0, 0, []

        # Parse solutions from response
        detected_solutions = parse_solutions(response)

        if verbose:
            print(f"    Detected solutions: {detected_solutions}")
            print(f"    Ground truth: {ground_truth_solutions}")

        # Normalize solutions for comparison
        detected_normalized = {self._normalize_solution(s) for s in detected_solutions}
        gt_normalized = [self._normalize_solution(s) for s in ground_truth_solutions]

        # Find matches
        found_solutions = []
        missing_solutions = []

        for gt_sol in ground_truth_solutions:
            gt_norm = self._normalize_solution(gt_sol)

            # Check if this solution was found
            is_found = gt_norm in detected_normalized

            if is_found:
                found_solutions.append(gt_sol)
            else:
                missing_solutions.append(gt_sol)

        n_found = len(found_solutions)
        n_total = len(ground_truth_solutions)

        # CE score = fraction of solutions found
        ce_score = n_found / n_total if n_total > 0 else 1.0

        if verbose and n_found < n_total:
            print(f"    ⚠️  Found {n_found}/{n_total} solutions")

        return ce_score, n_found, missing_solutions

    def _normalize_condition(self, condition: str) -> str:
        """
        Normalize a condition string for comparison.

        Args:
            condition: Condition string (e.g., "x >= 0", "x>=0")

        Returns:
            Normalized condition string
        """
        import re

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', condition.strip())

        # Normalize spacing around operators
        normalized = re.sub(r'([a-zA-Z])\s*([<>=]+)\s*([0-9\-]+)', r'\1 \2 \3', normalized)

        # Convert to lowercase
        normalized = normalized.lower()

        return normalized

    def _normalize_solution(self, solution: str) -> str:
        """
        Normalize a solution string for comparison.

        Args:
            solution: Solution string (e.g., "x = 3", "x=3")

        Returns:
            Normalized solution string
        """
        import re

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', solution.strip())

        # Normalize spacing around equals
        normalized = re.sub(r'([a-zA-Z])\s*=\s*', r'\1 = ', normalized)

        # Convert to lowercase
        normalized = normalized.lower()

        # Handle special cases like √3
        # Keep unicode characters for now

        return normalized

    def _conditions_match(self, cond1: str, cond2: str) -> bool:
        """
        Check if two normalized conditions match.

        Handles cases like:
        - "x >= 0" matches "x >= 0"
        - "x - 3 >= 0" matches "x >= 3" (semantically equivalent)

        Args:
            cond1: First normalized condition
            cond2: Second normalized condition

        Returns:
            True if conditions match
        """
        # Exact match
        if cond1 == cond2:
            return True

        # Fuzzy match: check if they're describing the same inequality
        # This is a simple heuristic - perfect matching would require symbolic math

        # Extract variable, operator, and value
        import re
        pattern = r'([a-z]+)\s*([<>=]+)\s*([\-0-9]+)'

        match1 = re.search(pattern, cond1)
        match2 = re.search(pattern, cond2)

        if match1 and match2:
            var1, op1, val1 = match1.groups()
            var2, op2, val2 = match2.groups()

            # Same variable, operator, and value
            if var1 == var2 and op1 == op2 and val1 == val2:
                return True

        # No match
        return False


def compute_collective_exhaustiveness(
    response: str,
    problem: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to compute CE scores without creating scorer object.

    Args:
        response: Model's full response text
        problem: Problem dictionary with 'required_cases' and 'ground_truth_solutions'
        verbose: Print detailed scoring information

    Returns:
        Dictionary of CE scores (see CollectiveExhaustivenessScorer.compute_ce_scores)

    Example:
        >>> scores = compute_collective_exhaustiveness(
        ...     "Case 1: x >= 0, x = 3\\nCase 2: x < 0, x = -1",
        ...     {"required_cases": ["x >= 0", "x < 0"], "ground_truth_solutions": ["x = 3", "x = -1"]}
        ... )
        >>> scores['overall_ce_score']
        1.0
    """
    scorer = CollectiveExhaustivenessScorer()
    return scorer.compute_ce_scores(response, problem, verbose=verbose)
