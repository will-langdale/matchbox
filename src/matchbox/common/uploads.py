"""Common objects for upload."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS


class BackendUploadType(StrEnum):
    """Enumeration of supported backend upload types."""

    INDEX = "index"
    RESULTS = "results"

    @property
    def schema(self):
        """Get the schema for the upload type."""
        return {
            BackendUploadType.INDEX: SCHEMA_INDEX,
            BackendUploadType.RESULTS: SCHEMA_RESULTS,
        }[self]


class UploadStatus(BaseModel):
    """Response model for any file upload processes."""

    id: str
    stage: Literal[
        "ready", "awaiting_upload", "queued", "processing", "complete", "failed"
    ]
    update_timestamp: datetime
    details: str | None = None
    entity: BackendUploadType | None = None

    _status_code_mapping = {
        "ready": 200,
        "complete": 200,
        "failed": 400,
        "awaiting_upload": 202,
        "queued": 200,
        "processing": 200,
    }

    def get_http_code(self, stage: str) -> int:
        """Get the HTTP status code for the upload stage."""
        if self.stage == "failed":
            return 400
        return self._status_code_mapping[self.stage]

    @classmethod
    def status_400_examples(cls) -> dict:
        """Examples for 400 stage code."""
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "expired_id": {
                            "summary": "Upload ID expired",
                            "value": cls(
                                id="example_id",
                                stage="failed",
                                details=(
                                    "Upload ID not found or expired. Entries expire "
                                    "after 30 minutes of inactivity, including "
                                    "failed processes."
                                ),
                                entity=BackendUploadType.INDEX,
                                update_timestamp=datetime.now(),
                            ).model_dump(),
                        },
                        "schema_mismatch": {
                            "summary": "Schema validation error",
                            "value": cls(
                                id="example_id",
                                stage="failed",
                                details="Schema mismatch. Expected: ... Got: ...",
                                entity=BackendUploadType.INDEX,
                                update_timestamp=datetime.now(),
                            ).model_dump(),
                        },
                    },
                }
            }
        }
