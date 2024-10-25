from enum import StrEnum
from typing import Any

from pandas import DataFrame
from pydantic import BaseModel

from matchbox.common.exceptions import MatchboxModelError
from matchbox.common.results import ProbabilityResults
from matchbox.models.dedupers.base import Deduper
from matchbox.models.linkers.base import Linker
from matchbox.server import MatchboxDBAdapter, inject_backend
from matchbox.server.base import MatchboxModelAdapter
from matchbox.server.models import Probability


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

    def insert_probabilities(self, probabilities: list[Probability]) -> None:
        """Insert probabilities assocaited with the model into the backend database."""
        if not self._model:
            self._connect()

        self._model.insert_probabilities(
            probabilities=probabilities,
            batch_size=self._backend.settings.batch_size,
        )

    def set_truth_threshold(self, probability: float) -> None:
        """Set the truth threshold for the model."""
        if not self._model:
            self._connect()

        self._model.set_truth_threshold(probability=probability)

    def run(self) -> ProbabilityResults:
        """Execute the model and return results."""
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
