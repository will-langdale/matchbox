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


class BackendUploadType(StrEnum):
    INDEX = "index"
    RESULTS = "results"


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class CountResult(BaseModel):
    """Response model for count results"""

    entities: dict[BackendCountableType, int]


class UploadStatus(BaseModel):
    """Response model for any file upload processes, like Source or Model results."""

    id: str | None = None
    status: Literal["ready", "awaiting_upload", "processing", "complete", "failed"]
    details: str | None = None
    entity: BackendUploadType

    @classmethod
    def example_400_response_body(cls) -> dict:
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "expired_id": {
                            "summary": "Upload ID expired",
                            "value": cls(
                                id="example_id",
                                status="failed",
                                details=(
                                    "Upload ID not found or expired. Entries expire "
                                    "after 30 minutes of inactivity, including "
                                    "failed processes."
                                ),
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                        "schema_mismatch": {
                            "summary": "Schema validation error",
                            "value": cls(
                                id="example_id",
                                status="failed",
                                details="Schema mismatch. Expected: ... Got: ...",
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                    }
                }
            }
        }


class NotFoundError(BaseModel):
    """API error for a 404 status code"""

    details: str
    entity: BackendRetrievableType

    @classmethod
    def example_response_body(cls) -> dict:
        return {
            "content": {
                "application/json": {
                    "example": cls(
                        details="details", entity=BackendRetrievableType.SOURCE
                    ).model_dump()
                }
            }
        }
