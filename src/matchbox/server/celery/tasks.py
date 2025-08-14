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
TRACKER = RedisUploadTracker(SETTINGS.redis_uri, SETTINGS.uploads_expiry_minutes)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

app = Celery("matchbox", broker=SETTINGS.redis_uri)


@app.task(ignore_result=True)
def process_upload(
    upload_type: str,
    resolution_name: str,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Worker task to process uploaded file."""
    TRACKER.update(upload_id, "processing")
    client = BACKEND.settings.datastore.get_client()
    upload = TRACKER.get(upload_id)

    try:
        data = pa.Table.from_batches(
            [
                batch
                for batch in s3_to_recordbatch(
                    client=client, bucket=bucket, key=filename
                )
            ]
        )

        if upload.status.entity == BackendUploadType.INDEX:
            BACKEND.index(source_config=upload.metadata, data_hashes=data)
        elif upload.status.entity == BackendUploadType.RESULTS:
            BACKEND.set_model_results(name=upload.metadata.name, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.status.entity}")

        TRACKER.update(upload_id, "complete")
        logger.info("Completed ")

    except Exception as e:
        error_context = {
            "upload_id": upload_id,
            "upload_type": upload_type,
            "resolution_name": resolution_name,
            "bucket": bucket,
            "key": filename,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )
        details = (
            f"Error: {e}. Context: "
            f"Upload type: {getattr(upload.status, 'entity', 'unknown')}, "
            f"SourceConfig: {getattr(upload, 'metadata', 'unknown')}"
        )
        TRACKER.update(
            upload_id,
            "failed",
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            client.delete_object(Bucket=bucket, Key=filename)
        except Exception as delete_error:
            logger.error(
                f"Failed to delete S3 file {bucket}/{filename}: {delete_error}"
            )
