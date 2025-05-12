"""Common operations to produce model evaluation scores."""

from itertools import combinations
from typing import TypeVar

T = TypeVar("T")


def sets_to_pairs(elements: list[T]) -> T:
    """Convert set of cluster elements to implied pair-wise edges."""
    return [pair for pairs in combinations(elements) for pair in pairs]


def leaf_overlap(eval_groups, model_groups):
    """Compare overlap between eval and model clusters."""
    ...


def roc(eval_pairs, model_pairs):
    """Compute ROC scores."""
    ...
