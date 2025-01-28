from enum import StrEnum

from pydantic import BaseModel

from matchbox.common.sources import SourceAddress


class BackendEntityType(StrEnum):
    DATASETS = "datasets"
    MODELS = "models"
    DATA = "data"
    CLUSTERS = "clusters"
    CREATES = "creates"
    MERGES = "merges"
    PROPOSES = "proposes"


class ModelResultsType(StrEnum):
    PROBABILITIES = "probabilities"
    CLUSTERS = "clusters"


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class CountResult(BaseModel):
    """Response model for count results"""

    entities: dict[BackendEntityType, int]


class QueryParams:
    model_config = {"extra": "forbid"}

    source_address: SourceAddress
    resolution_id: int | None = None
    threshold: float | None = None
    limit: int | None = None
