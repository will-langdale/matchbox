"""A metadata registry of uploads and their status."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generator

import pyarrow as pa
import redis
from fastapi import (
    UploadFile,
)
from pyarrow import compute as pq
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import (
    BackendUploadType,
    ModelConfig,
    UploadStatus,
)
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.sources import SourceConfig

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


class UploadEntry(BaseModel):
    """Metadata entry for upload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SourceConfig | ModelConfig
    upload_type: BackendUploadType
    update_timestamp: datetime
    status: UploadStatus


class UploadTracker:
    """A simple registry of uploaded metadata and processing status, backed by Redis."""

    def __init__(self, redis_url: str, expiry_minutes: int):
        """Connect Redis and initialise tracker object."""
        self.expiry_minutes = expiry_minutes
        self.redis = redis.Redis.from_url(redis_url)

    @staticmethod
    def _generate_id():
        return str(uuid.uuid4())

    def _add_entry(self, key: str, value: str):
        expiry_seconds = self.expiry_minutes * 60
        self.redis.setex(f"upload:{key}", expiry_seconds, value)

    def add_source(self, metadata: BaseModel) -> str:
        """Register source metadata and return ID."""
        upload_id = self._generate_id()

        entry = UploadEntry(
            metadata=metadata,
            upload_type=BackendUploadType.INDEX,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=upload_id, status="awaiting_upload", entity=BackendUploadType.INDEX
            ),
        )

        self._add_entry(upload_id, entry.model_dump_json())

        return upload_id

    def add_model(self, metadata: BaseModel) -> str:
        """Register model results metadata and return ID."""
        upload_id = self._generate_id()

        entry = UploadEntry(
            metadata=metadata,
            upload_type=BackendUploadType.RESULTS,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=upload_id, status="awaiting_upload", entity=BackendUploadType.RESULTS
            ),
        )

        self._add_entry(upload_id, entry.model_dump_json())

        return upload_id

    def get(self, upload_id: str) -> UploadEntry | None:
        """Retrieve metadata by ID if not expired."""
        data = self.redis.get(upload_id)
        if not data:
            return None

        entry = UploadEntry.model_validate_json(data)

        return entry

    def update_status(
        self, upload_id: str, status: str, details: str | None = None
    ) -> bool:
        """Update the status of an entry.

        Raises:
            KeyError: If entry not found.
        """
        data = self.redis.get(upload_id)
        if not data:
            raise KeyError(f"Entry {upload_id} not found.")

        entry = UploadEntry.model_validate_json(data)
        entry.status.status = status
        if details:
            entry.status.details = details
        entry.update_timestamp = datetime.now()

        self._add_entry(upload_id, entry.model_dump_json())

        return True

    def remove(self, upload_id: str) -> bool:
        """Remove an entry from the tracker."""
        return self.redis.delete(upload_id) == 1


def table_to_s3(
    client: S3Client,
    bucket: str,
    key: str,
    file: UploadFile,
    expected_schema: pa.Schema,
) -> str:
    """Upload a PyArrow Table to S3 and return the key.

    Args:
        client: The S3 client to use.
        bucket: The S3 bucket to upload to.
        key: The key to upload to.
        file: The file to upload.
        expected_schema: The schema that the file should match.

    Raises:
        MatchboxServerFileError: If the file is not a valid Parquet file or the schema
            does not match the expected schema.

    Returns:
        The key of the uploaded file.
    """
    try:
        table = pq.read_table(file.file)

        if not table.schema.equals(expected_schema):
            raise MatchboxServerFileError(
                message=(
                    "Schema mismatch. "
                    f"Expected:\n{expected_schema}\nGot:\n{table.schema}"
                )
            )

        file.file.seek(0)

        client.put_object(Bucket=bucket, Key=key, Body=file.file)

    except Exception as e:
        if isinstance(e, MatchboxServerFileError):
            raise
        raise MatchboxServerFileError(message=f"Invalid Parquet file: {str(e)}") from e

    return key


def s3_to_recordbatch(
    client: S3Client, bucket: str, key: str, batch_size: int = 1000
) -> Generator[pa.RecordBatch, None, None]:
    """Download a PyArrow Table from S3 and stream it as RecordBatches."""
    response = client.get_object(Bucket=bucket, Key=key)
    buffer = pa.BufferReader(response["Body"].read())

    parquet_file = pq.ParquetFile(buffer)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch
