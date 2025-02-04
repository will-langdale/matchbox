from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class BackendCountableType(StrEnum):
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


class BackendRetrievableType(StrEnum):
    SOURCE = "source"
    RESOLUTION = "resolution"


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class CountResult(BaseModel):
    """Response model for count results"""

    entities: dict[BackendCountableType, int]


class UploadStatus(BaseModel):
    """Response model for any file upload processes, like Source or Model results."""

    id: str | None
    status: Literal["ready", "awaiting_upload", "processing", "failed"]


class NotFoundError(BaseModel):
    """API error for a 404 status code"""

    details: str
    entity: BackendRetrievableType

    @classmethod
    def example_response_body(cls):
        return {
            "content": {
                "application/json": {
                    "example": cls(
                        details="details", entity=BackendRetrievableType.SOURCE
                    ).model_dump()
                }
            }
        }
