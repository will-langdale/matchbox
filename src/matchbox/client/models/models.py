from typing import Any, ParamSpec, TypeVar

from pandas import DataFrame

from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.results import Results
from matchbox.common.dtos import ModelAncestor, ModelMetadata, ModelType
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.server import MatchboxDBAdapter, inject_backend

P = ParamSpec("P")
R = TypeVar("R")


class Model:
    """Unified model class for both linking and deduping operations."""

    def __init__(
        self,
        metadata: ModelMetadata,
        model_instance: Linker | Deduper,
        left_data: DataFrame,
        right_data: DataFrame | None = None,
    ):
        self.metadata = metadata
        self.model_instance = model_instance
        self.left_data = left_data
        self.right_data = right_data

    @inject_backend
    def insert_model(self, backend: MatchboxDBAdapter) -> None:
        """Insert the model into the backend database."""
        backend.insert_model(model=self.metadata)

    @property
    @inject_backend
    def results(self, backend: MatchboxDBAdapter) -> Results:
        """Retrieve results associated with the model from the database."""
        results = backend.get_model_results(model=self.metadata.name)
        return Results(probabilities=results, metadata=self.metadata)

    @results.setter
    @inject_backend
    def results(self, backend: MatchboxDBAdapter, results: Results) -> None:
        """Write results associated with the model to the database."""
        backend.set_model_results(
            model=self.metadata.name, results=results.probabilities
        )

    @property
    @inject_backend
    def truth(self, backend: MatchboxDBAdapter) -> float:
        """Retrieve the truth threshold for the model."""
        return backend.get_model_truth(model=self.metadata.name)

    @truth.setter
    @inject_backend
    def truth(self, backend: MatchboxDBAdapter, truth: float) -> None:
        """Set the truth threshold for the model."""
        backend.set_model_truth(model=self.metadata.name, truth=truth)

    @property
    @inject_backend
    def ancestors(self, backend: MatchboxDBAdapter) -> dict[str, float]:
        """Retrieve the ancestors of the model."""
        return {
            ancestor.name: ancestor.truth
            for ancestor in backend.get_model_ancestors(model=self.metadata.name)
        }

    @property
    @inject_backend
    def ancestors_cache(self, backend: MatchboxDBAdapter) -> dict[str, float]:
        """Retrieve the ancestors cache of the model."""
        return {
            ancestor.name: ancestor.truth
            for ancestor in backend.get_model_ancestors_cache(model=self.metadata.name)
        }

    @ancestors_cache.setter
    @inject_backend
    def ancestors_cache(
        self, backend: MatchboxDBAdapter, ancestors_cache: dict[str, float]
    ) -> None:
        """Set the ancestors cache of the model."""
        backend.set_model_ancestors_cache(
            model=self.metadata.name,
            ancestors_cache=[
                ModelAncestor(name=k, truth=v) for k, v in ancestors_cache
            ],
        )

    def run(self) -> Results:
        """Execute the model pipeline and return results."""
        if self.metadata.type == ModelType.LINKER:
            if self.right_data is None:
                raise MatchboxResolutionNotFoundError(
                    "Right dataset required for linking"
                )

            results = self.model_instance.link(
                left=self.left_data, right=self.right_data
            )
        else:
            results = self.model_instance.dedupe(data=self.left_data)

        return Results(
            probabilities=results,
            model=self,
            metadata=self.metadata,
        )


def make_model(
    model_name: str,
    description: str,
    model_class: type[Linker] | type[Deduper],
    model_settings: dict[str, Any],
    left_data: DataFrame,
    left_resolution: str,
    right_data: DataFrame | None = None,
    right_resolution: str | None = None,
) -> Model:
    """Create a unified model instance for either linking or deduping operations.

    Args:
        model_name: Your unique identifier for the model
        description: Description of the model run
        model_class: Either Linker or Deduper class
        model_settings: Configuration settings for the model
        left_data: Primary dataset
        left_resolution: Resolution name for primary model or dataset
        right_data: Secondary dataset (linking only)
        right_resolution: Resolution name for secondary model or dataset (linking only)

    Returns:
        Model: Configured model instance ready for execution
    """
    model_type = (
        ModelType.LINKER if issubclass(model_class, Linker) else ModelType.DEDUPER
    )

    if model_type == ModelType.LINKER and (
        right_data is None or right_resolution is None
    ):
        raise ValueError("Linking requires both right_data and right_resolution")

    model_instance = model_class.from_settings(**model_settings)

    if model_type == ModelType.LINKER:
        model_instance.prepare(left=left_data, right=right_data)
    else:
        model_instance.prepare(data=left_data)

    metadata = ModelMetadata(
        name=model_name,
        description=description,
        type=model_type.value,
        left_resolution=left_resolution,
        right_resolution=right_resolution,
    )

    return Model(
        metadata=metadata,
        model_instance=model_instance,
        left_data=left_data,
        right_data=right_data,
    )
