"""Baseline compression methods for comparison."""

from .full_context import FullContextBaseline
from .random_tokens import RandomTokenBaseline
from .no_context import NoContextBaseline

__all__ = [
    "FullContextBaseline",
    "RandomTokenBaseline",
    "NoContextBaseline",
]
