"""
Mutual Exclusivity (ME) metrics for MECE reasoning evaluation.

These metrics measure how well reasoning steps avoid overlap and redundancy.
Higher scores indicate better mutual exclusivity.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from .parsers import (
    parse_reasoning_steps,
    extract_case_conditions,
    count_overlapping_conditions
)


class MutualExclusivityScorer:
    """
    Compute mutual exclusivity scores for reasoning chains.

    Uses two complementary approaches:
    1. Embedding similarity - measures semantic overlap between steps
    2. Case condition overlap - detects mathematical overlap in conditions

    Example:
        >>> scorer = MutualExclusivityScorer()
        >>> response = "Step 1: When x >= 0, solve x = 2x - 3\\nStep 2: When x < 0, solve -x = 2x - 3"
        >>> scores = scorer.compute_me_scores(response)
        >>> print(scores['overall_me_score'])
        0.95  # High score = good mutual exclusivity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        cache_model: bool = True
    ):
        """
        Initialize ME scorer with embedding model.

        Args:
            model_name: SentenceTransformer model to use for embeddings
            similarity_threshold: Threshold above which steps are considered overlapping
            cache_model: Whether to cache the model in memory

        Raises:
            ImportError: If sentence-transformers not installed
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.cache_model = cache_model
        self._model: Optional[SentenceTransformer] = None

        if cache_model:
            self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise ImportError(
                    f"Failed to load SentenceTransformer model: {e}\n"
                    "Install with: pip install sentence-transformers"
                )

    def compute_me_scores(
        self,
        response: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compute all mutual exclusivity scores for a response.

        Args:
            response: Model's full response text
            verbose: Print detailed scoring information

        Returns:
            Dictionary containing:
            - step_similarity_me_score: Embedding-based ME score (0-1)
            - case_overlap_me_score: Condition overlap ME score (0-1)
            - overall_me_score: Combined ME score (0-1)
            - n_steps: Number of reasoning steps detected
            - n_conditions: Number of case conditions detected
            - n_overlapping_pairs: Number of overlapping condition pairs
            - avg_pairwise_similarity: Average embedding similarity

        Example:
            >>> scorer = MutualExclusivityScorer()
            >>> scores = scorer.compute_me_scores(
            ...     "Step 1: For x >= 0, x = 3\\nStep 2: For x < 0, x = -1"
            ... )
            >>> scores['overall_me_score']
            0.92
        """
        # Parse reasoning steps
        steps = parse_reasoning_steps(response)
        n_steps = len(steps)

        if verbose:
            print(f"\n📊 Mutual Exclusivity Scoring")
            print(f"  Steps detected: {n_steps}")
            if n_steps > 0:
                for i, step in enumerate(steps, 1):
                    print(f"    {i}. {step[:60]}...")

        # Edge case: < 2 steps means perfect ME (no overlap possible)
        if n_steps < 2:
            return {
                'step_similarity_me_score': 1.0,
                'case_overlap_me_score': 1.0,
                'overall_me_score': 1.0,
                'n_steps': n_steps,
                'n_conditions': 0,
                'n_overlapping_pairs': 0,
                'avg_pairwise_similarity': 0.0,
            }

        # Compute embedding-based ME score
        step_similarity_score, avg_similarity = self._compute_embedding_me(
            steps, verbose=verbose
        )

        # Compute case condition overlap ME score
        case_overlap_score, n_conditions, n_overlaps = self._compute_case_overlap_me(
            steps, verbose=verbose
        )

        # Combine scores (weighted average)
        # Weight embedding similarity more heavily as it's more reliable
        overall_score = 0.7 * step_similarity_score + 0.3 * case_overlap_score

        results = {
            'step_similarity_me_score': round(step_similarity_score, 4),
            'case_overlap_me_score': round(case_overlap_score, 4),
            'overall_me_score': round(overall_score, 4),
            'n_steps': n_steps,
            'n_conditions': n_conditions,
            'n_overlapping_pairs': n_overlaps,
            'avg_pairwise_similarity': round(avg_similarity, 4),
        }

        if verbose:
            print(f"\n  Results:")
            print(f"    Embedding ME score: {results['step_similarity_me_score']:.3f}")
            print(f"    Case overlap ME score: {results['case_overlap_me_score']:.3f}")
            print(f"    Overall ME score: {results['overall_me_score']:.3f}")

        return results

    def _compute_embedding_me(
        self,
        steps: List[str],
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Compute ME score based on embedding similarity.

        High similarity between steps indicates poor mutual exclusivity.

        Args:
            steps: List of reasoning step strings
            verbose: Print detailed information

        Returns:
            Tuple of (me_score, avg_pairwise_similarity)
            - me_score: 0-1, higher = better ME
            - avg_pairwise_similarity: Average cosine similarity
        """
        if len(steps) < 2:
            return 1.0, 0.0

        # Load model if needed
        self._load_model()

        # Compute embeddings
        try:
            embeddings = self._model.encode(steps, convert_to_tensor=False)
            embeddings = np.array(embeddings)
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Embedding computation failed: {e}")
            # Fallback: assume moderate ME
            return 0.5, 0.5

        # Compute pairwise cosine similarities
        similarities = []
        n = len(embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        if not similarities:
            return 1.0, 0.0

        avg_similarity = float(np.mean(similarities))

        # ME score: inverse of similarity
        # High similarity = low ME
        # Use sigmoid-like transformation for smoother scores
        me_score = 1.0 - avg_similarity

        # Clip to [0, 1]
        me_score = max(0.0, min(1.0, me_score))

        if verbose:
            print(f"    Avg pairwise similarity: {avg_similarity:.3f}")
            max_sim = max(similarities)
            print(f"    Max similarity: {max_sim:.3f}")
            if max_sim > self.similarity_threshold:
                print(f"    ⚠️  High overlap detected (>{self.similarity_threshold})")

        return me_score, avg_similarity

    def _compute_case_overlap_me(
        self,
        steps: List[str],
        verbose: bool = False
    ) -> Tuple[float, int, int]:
        """
        Compute ME score based on case condition overlap.

        Detects mathematical conditions (e.g., "x >= 0", "x < 0") and checks
        for overlaps like "x >= 0" and "x > 0".

        Args:
            steps: List of reasoning step strings
            verbose: Print detailed information

        Returns:
            Tuple of (me_score, n_conditions, n_overlaps)
            - me_score: 0-1, higher = better ME
            - n_conditions: Number of conditions detected
            - n_overlaps: Number of overlapping condition pairs
        """
        # Extract case conditions
        conditions = extract_case_conditions(steps)
        n_conditions = len(conditions)

        if verbose and n_conditions > 0:
            print(f"    Conditions detected: {n_conditions}")
            for cond in conditions:
                print(f"      - {cond}")

        # Edge case: < 2 conditions means perfect ME
        if n_conditions < 2:
            return 1.0, n_conditions, 0

        # Count overlaps
        n_overlaps = count_overlapping_conditions(conditions)

        # Compute ME score
        # Total possible pairs
        total_pairs = n_conditions * (n_conditions - 1) // 2

        # ME score = 1 - (fraction of overlapping pairs)
        me_score = 1.0 - (n_overlaps / total_pairs)

        # Clip to [0, 1]
        me_score = max(0.0, min(1.0, me_score))

        if verbose and n_overlaps > 0:
            print(f"    ⚠️  Overlapping condition pairs: {n_overlaps}/{total_pairs}")

        return me_score, n_conditions, n_overlaps


def compute_mutual_exclusivity(
    response: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to compute ME scores without creating scorer object.

    Args:
        response: Model's full response text
        model_name: SentenceTransformer model to use
        verbose: Print detailed scoring information

    Returns:
        Dictionary of ME scores (see MutualExclusivityScorer.compute_me_scores)

    Example:
        >>> scores = compute_mutual_exclusivity(
        ...     "Step 1: When x >= 0\\nStep 2: When x < 0"
        ... )
        >>> scores['overall_me_score']
        0.92
    """
    scorer = MutualExclusivityScorer(model_name=model_name, cache_model=False)
    return scorer.compute_me_scores(response, verbose=verbose)
