"""Module implementing client-side evaluation features."""

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.client.eval import EvalData, compare_models, get_samples

__all__ = ["EvalData", "compare_models", "get_samples", "EntityResolutionApp"]
