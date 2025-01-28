from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


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


class SourceStatus(BaseModel):
    """Response model for Source status"""

    status: Literal["ready", "indexing", "failed"]
