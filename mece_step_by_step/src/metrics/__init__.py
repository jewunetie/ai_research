"""
MECE metrics package for evaluating reasoning quality.

This package provides computational metrics for measuring:
- Mutual Exclusivity (ME): How well steps avoid overlap
- Collective Exhaustiveness (CE): How well all cases are covered
- Accuracy: How correct the final solutions are
"""

# Always available - no heavy dependencies
from .parsers import (
    parse_reasoning_steps,
    extract_case_conditions,
    count_overlapping_conditions,
    parse_solutions,
    normalize_solution,
)

# Accuracy - no heavy dependencies
from .accuracy import (
    AccuracyScorer,
    compute_accuracy,
)

# Try to import ME and CE metrics (require numpy/sentence-transformers)
try:
    from .mutual_exclusivity import (
        MutualExclusivityScorer,
        compute_mutual_exclusivity,
    )
    _has_me_metrics = True
except ImportError:
    _has_me_metrics = False
    MutualExclusivityScorer = None
    compute_mutual_exclusivity = None

try:
    from .collective_exhaustiveness import (
        CollectiveExhaustivenessScorer,
        compute_collective_exhaustiveness,
    )
    _has_ce_metrics = True
except ImportError:
    _has_ce_metrics = False
    CollectiveExhaustivenessScorer = None
    compute_collective_exhaustiveness = None

__all__ = [
    # Parsers
    'parse_reasoning_steps',
    'extract_case_conditions',
    'count_overlapping_conditions',
    'parse_solutions',
    'normalize_solution',
    # Mutual Exclusivity (may be None if dependencies not installed)
    'MutualExclusivityScorer',
    'compute_mutual_exclusivity',
    # Collective Exhaustiveness (may be None if dependencies not installed)
    'CollectiveExhaustivenessScorer',
    'compute_collective_exhaustiveness',
    # Accuracy
    'AccuracyScorer',
    'compute_accuracy',
]
