"""Worker logic to process user uploads."""

from typing import TYPE_CHECKING, Any

import pyarrow as pa
from celery import Celery

from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
from matchbox.common.uploads import BackendUploadType
from matchbox.server.base import (
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)
from matchbox.server.uploads import RedisUploadTracker, s3_to_recordbatch

SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
SETTINGS: MatchboxServerSettings = SettingsClass()
BACKEND = settings_to_backend(SETTINGS)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

app = Celery("matchbox", broker=SETTINGS.redis_uri)
tracker = RedisUploadTracker(SETTINGS.redis_uri, SETTINGS.uploads_expiry_minutes)


@app.task(ignore_result=True)
def process_upload(
    upload_id: str,
    bucket: str,
    key: str,
) -> None:
    """Worker task to process uploaded file."""
    tracker.update(upload_id, "processing")
    client = BACKEND.settings.datastore.get_client()
    upload = tracker.get(upload_id)

    try:
        data = pa.Table.from_batches(
            [
                batch
                for batch in s3_to_recordbatch(client=client, bucket=bucket, key=key)
            ]
        )

        if upload.upload_type == BackendUploadType.INDEX:
            BACKEND.index(source_config=upload.metadata, data_hashes=data)
        elif upload.upload_type == BackendUploadType.RESULTS:
            BACKEND.set_model_results(name=upload.metadata.name, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.upload_type}")

        tracker.update(upload_id, "complete")

    except Exception as e:
        error_context = {
            "upload_id": upload_id,
            "upload_type": getattr(upload, "upload_type", "unknown"),
            "metadata": getattr(upload, "metadata", "unknown"),
            "bucket": bucket,
            "key": key,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )
        details = (
            f"Error: {e}. Context: "
            f"Upload type: {getattr(upload, 'upload_type', 'unknown')}, "
            f"SourceConfig: {getattr(upload, 'metadata', 'unknown')}"
        )
        tracker.update(
            upload_id,
            "failed",
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except Exception as delete_error:
            logger.error(f"Failed to delete S3 file {bucket}/{key}: {delete_error}")
