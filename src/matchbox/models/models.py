from enum import StrEnum
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import numpy as np
from pandas import DataFrame, Series
from pydantic import BaseModel

from matchbox.common.exceptions import MatchboxModelError
from matchbox.common.results import (
    ClusterResults,
    ProbabilityResults,
    Results,
    to_clusters,
)
from matchbox.models.dedupers.base import Deduper
from matchbox.models.linkers.base import Linker
from matchbox.server import MatchboxDBAdapter, inject_backend
from matchbox.server.base import MatchboxModelAdapter

P = ParamSpec("P")
R = TypeVar("R")


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    name: str
    description: str
    type: ModelType
    left_source: str
    right_source: str | None = None  # Only used for linker models


def ensure_connection(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to ensure model connection before method execution."""

    @wraps(func)
    def wrapper(self: "Model", *args: P.args, **kwargs: P.kwargs) -> R:
        if not self._model:
            self._connect()
        return func(self, *args, **kwargs)

    return wrapper


class Model:
    """Unified model class for both linking and deduping operations."""

    def __init__(
        self,
        metadata: ModelMetadata,
        model_instance: Linker | Deduper,
        left_data: DataFrame,
        right_data: DataFrame | None = None,
        backend: MatchboxDBAdapter | None = None,
    ):
        self.metadata = metadata
        self.model_instance = model_instance
        self.left_data = left_data
        self.right_data = right_data
        self._backend = backend
        self._model: MatchboxModelAdapter | None = None

    def _connect(self) -> None:
        """Establish connection to the model in the backend database."""
        if not self._backend:
            raise MatchboxModelError("No backend configured for this model")

        try:
            self._model = self._backend.get_model(self.metadata.name)
        except Exception as e:
            raise MatchboxModelError from e

    def insert_model(self) -> None:
        """Insert the model into the backend database."""
        if not self._backend:
            raise MatchboxModelError("No backend configured for this model")

        try:
            self._backend.insert_model(
                model=self.metadata.name,
                left=self.metadata.left_source,
                right=self.metadata.right_source,
                description=self.metadata.description,
            )
            self._connect()
        except Exception as e:
            raise MatchboxModelError from e

    @property
    @ensure_connection
    def probabilities(self) -> ProbabilityResults:
        """Retrieve probabilities associated with the model from the database."""
        n = len(self._model.probabilities)

        left_arr = np.empty(n, dtype="object")
        right_arr = np.empty(n, dtype="object")
        prob_arr = np.empty(n, dtype="float64")

        for i, prob in enumerate(self._model.probabilities):
            left_arr[i] = prob.left
            right_arr[i] = prob.right
            prob_arr[i] = prob.probability

        df = DataFrame(
            {
                "left_id": Series(left_arr, dtype="binary[pyarrow]"),
                "right_id": Series(right_arr, dtype="binary[pyarrow]"),
                "probability": Series(prob_arr, dtype="float64[pyarrow]"),
            }
        )
        return ProbabilityResults(dataframe=df, model=self)

    @property
    @ensure_connection
    def clusters(self) -> ClusterResults:
        """Retrieve clusters associated with the model from the database."""
        total_rows = sum(len(cluster.children) for cluster in self._model.clusters)

        parent_arr = np.empty(total_rows, dtype="object")
        child_arr = np.empty(total_rows, dtype="object")
        threshold_arr = np.empty(total_rows, dtype="float64")

        idx = 0
        for cluster in self._model.clusters:
            n_children = len(cluster.children)
            # Set parent, repeated for each child
            parent_arr[idx : idx + n_children] = cluster.parent
            # Set children
            child_arr[idx : idx + n_children] = cluster.children
            # Set threshold, repeated for each child)
            threshold_arr[idx : idx + n_children] = cluster.threshold
            idx += n_children

        df = DataFrame(
            {
                "parent": Series(parent_arr, dtype="binary[pyarrow]"),
                "child": Series(child_arr, dtype="binary[pyarrow]"),
                "threshold": Series(threshold_arr, dtype="float64[pyarrow]"),
            }
        )
        return ClusterResults(dataframe=df, model=self)

    @clusters.setter
    @ensure_connection
    def clusters(self, clusters: ClusterResults) -> None:
        """Insert clusters associated with the model into the backend database."""
        self._model.clusters = clusters.to_records(backend=self._backend)

    @property
    @ensure_connection
    def truth(self) -> float:
        """Retrieve the truth threshold for the model."""
        return self._model.truth

    @truth.setter
    @ensure_connection
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        self._model.truth = truth

    @property
    @ensure_connection
    def ancestors(self) -> dict[str, float]:
        """Retrieve the ancestors of the model."""
        return self._model.ancestors

    @property
    @ensure_connection
    def ancestors_cache(self) -> dict[str, float]:
        """Retrieve the ancestors cache of the model."""
        return self._model.ancestors_cache

    @ancestors_cache.setter
    @ensure_connection
    def ancestors_cache(self, ancestors_cache: dict[str, float]) -> None:
        """Set the ancestors cache of the model."""
        self._model.ancestors_cache = ancestors_cache

    def calculate_probabilities(self) -> ProbabilityResults:
        """Calculate probabilities for the model."""
        if self.metadata.type == ModelType.LINKER:
            if self.right_data is None:
                raise MatchboxModelError("Right dataset required for linking")

            results = self.model_instance.link(
                left=self.left_data, right=self.right_data
            )
        else:
            results = self.model_instance.dedupe(data=self.left_data)

        return ProbabilityResults(
            dataframe=results,
            model=self,
            description=self.metadata.description,
            left=self.metadata.left_source,
            right=self.metadata.right_source or self.metadata.left_source,
        )

    def calculate_clusters(self, probabilities: ProbabilityResults) -> ClusterResults:
        """Calculate clusters for the model based on probabilities."""
        return to_clusters(results=probabilities)

    def run(self) -> Results:
        """Execute the model pipeline and return results."""
        probabilities = self.calculate_probabilities()
        clusters = self.calculate_clusters(probabilities)

        return Results(model=self, probabilities=probabilities, clusters=clusters)


@inject_backend
def make_model(
    backend: MatchboxDBAdapter,
    model_name: str,
    description: str,
    model_class: type[Linker] | type[Deduper],
    model_settings: dict[str, Any],
    left_data: DataFrame,
    left_source: str,
    right_data: DataFrame | None = None,
    right_source: str | None = None,
) -> Model:
    """
    Create a unified model instance for either linking or deduping operations.

    Args:
        model_name: Your unique identifier for the model
        description: Description of the model run
        model_class: Either Linker or Deduper class
        model_settings: Configuration settings for the model
        data: Primary dataset
        data_source: Source identifier for the primary dataset
        right_data: Secondary dataset (only for linking)
        right_source: Source identifier for secondary dataset (only for linking)
        backend: Optional MatchboxDBAdapter instance

    Returns:
        Model: Configured model instance ready for execution
    """
    model_type = (
        ModelType.LINKER if issubclass(model_class, Linker) else ModelType.DEDUPER
    )

    if model_type == ModelType.LINKER and (right_data is None or right_source is None):
        raise ValueError("Linking requires both right_data and right_source")

    model_instance = model_class.from_settings(**model_settings)

    if model_type == ModelType.LINKER:
        model_instance.prepare(left=left_data, right=right_data)
    else:
        model_instance.prepare(data=left_data)

    metadata = ModelMetadata(
        name=model_name,
        description=description,
        type=model_type.value,
        left_source=left_source,
        right_source=right_source,
    )

    return Model(
        metadata=metadata,
        model_instance=model_instance,
        left_data=left_data,
        right_data=right_data,
        backend=backend,
    )
