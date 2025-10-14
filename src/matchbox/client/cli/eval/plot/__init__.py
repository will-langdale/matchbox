"""Plot components for entity resolution evaluation tool."""

from matchbox.client.cli.eval.plot.core import (
    _deduplicate_recall_values,
    compute_pr_envelope,
    interpolate_pr_curve,
    plotext_pr_envelope,
)

__all__ = [
    "_deduplicate_recall_values",
    "compute_pr_envelope",
    "interpolate_pr_curve",
    "plotext_pr_envelope",
]
