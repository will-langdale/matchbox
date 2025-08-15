"""Worker logic to process user uploads."""

from functools import partial
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from celery import Celery

from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
from matchbox.common.uploads import BackendUploadType
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)
from matchbox.server.uploads import (
    UploadTracker,
    s3_to_recordbatch,
    upload_tracker_from_settings,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


CELERY_SETTINGS = get_backend_settings(MatchboxServerSettings().backend_type)()
CELERY_BACKEND: MatchboxDBAdapter | None = None
CELERY_TRACKER: UploadTracker | None = None

celery = Celery("matchbox", broker=CELERY_SETTINGS.redis_uri)
celery.conf.update(
    # Only acknowledge task (remove it from queue) after task completion
    task_acks_late=True,
    # Reduce pre-fetching (workers reserving tasks while they're still busy)
    # as it's not ideal for long-running tasks
    prefetch_multiplier=1,
)


def initialise_celery_worker():
    """Initialise backend and tracker for celery worker."""
    global CELERY_SETTINGS
    global CELERY_BACKEND
    global CELERY_TRACKER

    if not CELERY_BACKEND:
        CELERY_BACKEND = settings_to_backend(CELERY_SETTINGS)
    if not CELERY_TRACKER:
        CELERY_TRACKER = upload_tracker_from_settings(CELERY_SETTINGS)


def process_upload(
    backend: MatchboxDBAdapter,
    tracker: UploadTracker,
    s3_client: S3Client,
    upload_type: str,
    resolution_name: str,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Generic task to process uploaded file, usable by FastAPI BackgroundTasks."""
    tracker.update(upload_id, "processing")
    upload = tracker.get(upload_id)

    try:
        data = pa.Table.from_batches(
            [
                batch
                for batch in s3_to_recordbatch(
                    client=s3_client, bucket=bucket, key=filename
                )
            ]
        )

        if upload.status.entity == BackendUploadType.INDEX:
            backend.index(source_config=upload.metadata, data_hashes=data)
        elif upload.status.entity == BackendUploadType.RESULTS:
            backend.set_model_results(name=upload.metadata.name, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.status.entity}")

        tracker.update(upload_id, "complete")

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
        tracker.update(
            upload_id,
            "failed",
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            s3_client.delete_object(Bucket=bucket, Key=filename)
        except Exception as delete_error:
            logger.error(
                f"Failed to delete S3 file {bucket}/{filename}: {delete_error}"
            )


@celery.task(ignore_result=True)
def process_upload_celery(
    upload_type: str,
    resolution_name: str,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Celery task to process uploaded file, with only serialisable arguments."""
    initialise_celery_worker()

    partial(
        process_upload,
        backend=CELERY_BACKEND,
        tracker=CELERY_TRACKER,
        s3_client=CELERY_BACKEND.settings.datastore.get_client(),
    )(
        upload_type=upload_type,
        resolution_name=resolution_name,
        upload_id=upload_id,
        bucket=bucket,
        filename=filename,
    )
