"""
Accuracy metrics for evaluating solution correctness.

These metrics measure how well the model's solutions match the ground truth.
"""

from typing import List, Dict, Any, Set
from .parsers import parse_solutions, normalize_solution


class AccuracyScorer:
    """
    Compute accuracy scores for model solutions.

    Compares detected solutions against ground truth with multiple metrics:
    - Exact match accuracy
    - Partial credit (for multi-solution problems)
    - Precision and recall

    Example:
        >>> scorer = AccuracyScorer()
        >>> response = "The solutions are x = 3 and x = -1"
        >>> ground_truth = ["x = 3", "x = -1"]
        >>> scores = scorer.compute_accuracy(response, ground_truth)
        >>> print(scores['exact_match'])
        True
    """

    def __init__(
        self,
        normalize: bool = True,
        partial_credit: bool = True
    ):
        """
        Initialize accuracy scorer.

        Args:
            normalize: Whether to normalize solutions before comparison
            partial_credit: Whether to give partial credit for multi-solution problems
        """
        self.normalize = normalize
        self.partial_credit = partial_credit

    def compute_accuracy(
        self,
        response: str,
        ground_truth_solutions: List[str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compute all accuracy scores for a response.

        Args:
            response: Model's full response text
            ground_truth_solutions: List of correct solutions
            verbose: Print detailed scoring information

        Returns:
            Dictionary containing:
            - exact_match: Boolean, True if all solutions match exactly
            - partial_score: Float 0-1, fraction of solutions correct
            - precision: Float 0-1, fraction of detected solutions that are correct
            - recall: Float 0-1, fraction of ground truth solutions that were found
            - f1_score: Float 0-1, harmonic mean of precision and recall
            - n_correct: Number of correct solutions detected
            - n_incorrect: Number of incorrect solutions detected
            - n_total_gt: Total number of ground truth solutions
            - correct_solutions: List of correct solutions found
            - incorrect_solutions: List of incorrect solutions detected
            - missing_solutions: List of ground truth solutions not found

        Example:
            >>> scorer = AccuracyScorer()
            >>> scores = scorer.compute_accuracy(
            ...     "x = 3 and x = -1",
            ...     ["x = 3", "x = -1"]
            ... )
            >>> scores['exact_match']
            True
            >>> scores['f1_score']
            1.0
        """
        if verbose:
            print(f"\n📊 Accuracy Scoring")

        # Parse solutions from response
        detected_solutions = parse_solutions(response)

        # Normalize if requested
        if self.normalize:
            detected_normalized = {normalize_solution(s) for s in detected_solutions}
            gt_normalized = {normalize_solution(s) for s in ground_truth_solutions}
        else:
            detected_normalized = set(detected_solutions)
            gt_normalized = set(ground_truth_solutions)

        if verbose:
            print(f"  Detected solutions: {detected_solutions}")
            print(f"  Ground truth: {ground_truth_solutions}")
            if self.normalize:
                print(f"  Detected (normalized): {detected_normalized}")
                print(f"  GT (normalized): {gt_normalized}")

        # Find correct, incorrect, and missing solutions
        correct = detected_normalized & gt_normalized
        incorrect = detected_normalized - gt_normalized
        missing = gt_normalized - detected_normalized

        n_correct = len(correct)
        n_incorrect = len(incorrect)
        n_detected = len(detected_normalized)
        n_total_gt = len(gt_normalized)

        # Exact match: all GT solutions found, no incorrect ones
        exact_match = (n_correct == n_total_gt) and (n_incorrect == 0)

        # Partial score: fraction of GT solutions found
        partial_score = n_correct / n_total_gt if n_total_gt > 0 else 0.0

        # Precision: fraction of detected solutions that are correct
        precision = n_correct / n_detected if n_detected > 0 else 0.0

        # Recall: fraction of GT solutions that were found
        recall = n_correct / n_total_gt if n_total_gt > 0 else 0.0

        # F1 score: harmonic mean of precision and recall
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        results = {
            'exact_match': exact_match,
            'partial_score': round(partial_score, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'n_total_gt': n_total_gt,
            'correct_solutions': sorted(list(correct)),
            'incorrect_solutions': sorted(list(incorrect)),
            'missing_solutions': sorted(list(missing)),
        }

        if verbose:
            print(f"\n  Results:")
            print(f"    Exact match: {results['exact_match']}")
            print(f"    Partial score: {results['partial_score']:.3f}")
            print(f"    Precision: {results['precision']:.3f}")
            print(f"    Recall: {results['recall']:.3f}")
            print(f"    F1 score: {results['f1_score']:.3f}")
            if n_correct > 0:
                print(f"    ✓ Correct: {results['correct_solutions']}")
            if n_incorrect > 0:
                print(f"    ✗ Incorrect: {results['incorrect_solutions']}")
            if missing:
                print(f"    ⚠️  Missing: {results['missing_solutions']}")

        return results


def compute_accuracy(
    response: str,
    ground_truth_solutions: List[str],
    normalize: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to compute accuracy scores without creating scorer object.

    Args:
        response: Model's full response text
        ground_truth_solutions: List of correct solutions
        normalize: Whether to normalize solutions before comparison
        verbose: Print detailed scoring information

    Returns:
        Dictionary of accuracy scores (see AccuracyScorer.compute_accuracy)

    Example:
        >>> scores = compute_accuracy(
        ...     "The answer is x = 3",
        ...     ["x = 3"]
        ... )
        >>> scores['exact_match']
        True
    """
    scorer = AccuracyScorer(normalize=normalize)
    return scorer.compute_accuracy(response, ground_truth_solutions, verbose=verbose)
