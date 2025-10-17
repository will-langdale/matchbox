"""Public evaluation helpers for Matchbox clients."""

from matchbox.client.eval.samples import (  # noqa: F401
    DeduplicationResult,
    EvalData,
    EvaluationItem,
    ModelComparison,
    compare_models,
    create_display_dataframe,
    create_evaluation_item,
    deduplicate_columns,
    get_samples,
    precision_recall,
)

__all__ = [
    "DeduplicationResult",
    "EvalData",
    "EvaluationItem",
    "compare_models",
    "create_display_dataframe",
    "create_evaluation_item",
    "deduplicate_columns",
    "get_samples",
    "ModelComparison",
    "precision_recall",
]
