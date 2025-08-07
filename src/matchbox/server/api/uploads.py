"""A metadata registry of uploads and their status."""

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event, Thread

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import BackendUploadType, ModelConfig, UploadStatus
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig
from matchbox.server.api.arrow import s3_to_recordbatch
from matchbox.server.base import MatchboxDBAdapter


class UploadEntry(BaseModel):
    """Metadata entry for upload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SourceConfig | ModelConfig
    upload_type: BackendUploadType
    update_timestamp: datetime
    status: UploadStatus


class UploadTracker:
    """A simple in-memory registry of uploaded metadata and processing status.

    Uses a janitor pattern for lazy cleanup. Entries expire after a period of
    inactivity, where activity is defined as any update to the entry's status
    or access via the upload endpoint.
    """

    def __init__(self, expiry_minutes: int = 30):
        """Initialise the tracker with an expiry time in minutes."""
        self._tracker: dict[str, UploadEntry] = {}
        self.expiry_minutes = expiry_minutes

    def _is_expired(self, entry: UploadEntry) -> bool:
        """Check if an entry has expired due to inactivity."""
        return datetime.now() - entry.update_timestamp > timedelta(
            minutes=self.expiry_minutes
        )

    def _cleanup_if_needed(self) -> None:
        """Lazy cleanup - remove all expired entries."""
        expired = [id for id, entry in self._tracker.items() if self._is_expired(entry)]
        for id in expired:
            del self._tracker[id]

    def _update_timestamp(self, upload_id: str) -> None:
        """Update the timestamp on an entry to prevent expiry."""
        if entry := self._tracker.get(upload_id):
            entry.update_timestamp = datetime.now()

    def add_source(self, metadata: SourceConfig) -> str:
        """Register source metadata and return ID."""
        self._cleanup_if_needed()
        upload_id = str(uuid.uuid4())

        self._tracker[upload_id] = UploadEntry(
            metadata=metadata,
            upload_type=BackendUploadType.INDEX,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=upload_id, status="awaiting_upload", entity=BackendUploadType.INDEX
            ),
        )
        return upload_id

    def add_model(self, metadata: ModelConfig) -> str:
        """Register model results metadata and return ID."""
        self._cleanup_if_needed()
        upload_id = str(uuid.uuid4())

        self._tracker[upload_id] = UploadEntry(
            metadata=metadata,
            upload_type=BackendUploadType.RESULTS,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=upload_id,
                status="awaiting_upload",
                entity=BackendUploadType.RESULTS,
            ),
        )
        return upload_id

    def get(self, upload_id: str) -> UploadEntry | None:
        """Retrieve metadata by ID if not expired. Updates timestamp on access."""
        self._cleanup_if_needed()

        entry = self._tracker.get(upload_id)
        if not entry:
            return None

        if self._is_expired(entry):
            del self._tracker[upload_id]
            return None

        self._update_timestamp(upload_id)
        return entry

    def update_status(
        self, upload_id: str, status: str, details: str | None = None
    ) -> bool:
        """Update the status of an entry.

        Raises:
            KeyError: If entry not found.
        """
        if entry := self._tracker.get(upload_id):
            entry.status.status = status
            if details is not None:
                entry.status.details = details
            entry.update_timestamp = datetime.now()
            return True
        raise KeyError(f"Entry {upload_id} not found.")

    def remove(self, upload_id: str) -> bool:
        """Remove an entry from the tracker."""
        self._cleanup_if_needed()
        if upload_id in self._tracker:
            del self._tracker[upload_id]
            return True
        return False


@contextmanager
def heartbeat(
    upload_tracker: UploadTracker, upload_id: str, interval_seconds: float = 300
):
    """Context manager that updates status with a heartbeat.

    Args:
        upload_tracker: Tracker for updating status
        upload_id: ID of the upload being processed
        interval_seconds: How often to send heartbeat (default 5 minutes)
    """
    stop_event = Event()

    def _heartbeat():
        # Wait up to `interval_seconds`
        # If event is set, exit thread
        # If event is not set, update tracker
        while not stop_event.wait(interval_seconds):
            try:
                timestamp = datetime.now().isoformat()
                upload_tracker.update_status(
                    upload_id=upload_id,
                    status="processing",
                    details=f"Still processing... Last heartbeat: {timestamp}",
                )
            except Exception as e:
                logger.error(
                    f"Heartbeat for upload_id={upload_id} failed with error: {str(e)}"
                )

    thread = Thread(target=_heartbeat)
    thread.start()

    try:
        yield
    finally:
        stop_event.set()
        # Guarantees that heartbeat stops updating status before control is handed back
        # to main thread
        thread.join()


def process_upload(
    backend: MatchboxDBAdapter,
    upload_id: str,
    bucket: str,
    key: str,
    tracker: UploadTracker,
    heartbeat_seconds: int,
) -> None:
    """Background task to process uploaded file."""
    tracker.update_status(upload_id, "processing")
    client = backend.settings.datastore.get_client()
    upload = tracker.get(upload_id)

    try:
        with heartbeat(tracker, upload_id, interval_seconds=heartbeat_seconds):
            data = pa.Table.from_batches(
                [
                    batch
                    for batch in s3_to_recordbatch(
                        client=client, bucket=bucket, key=key
                    )
                ]
            )

            if upload.upload_type == BackendUploadType.INDEX:
                backend.index(source_config=upload.metadata, data_hashes=data)
            elif upload.upload_type == BackendUploadType.RESULTS:
                backend.set_model_results(name=upload.metadata.name, results=data)
            else:
                raise ValueError(f"Unknown upload type: {upload.upload_type}")

        tracker.update_status(upload_id, "complete")

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
        tracker.update_status(
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
