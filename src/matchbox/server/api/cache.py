import uuid
from datetime import datetime, timedelta
from enum import Enum

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.dtos import (
    BackendUploadType,
    UploadStatus,
)
from matchbox.common.sources import Source
from matchbox.server.api.arrow import s3_to_recordbatch
from matchbox.server.base import MatchboxDBAdapter


class MetadataSchema(Enum):
    index = SCHEMA_INDEX
    results = SCHEMA_RESULTS


class MetadataCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: Source
    upload_schema: MetadataSchema
    upload_type: BackendUploadType
    update_timestamp: datetime
    status: UploadStatus


class MetadataStore:
    """A simple in-memory cache of uploaded metadata and processing status.

    Uses a janitor pattern for lazy cleanup. Entries expire after a period of
    inactivity, where activity is defined as any update to the entry's status
    or access via the upload endpoint.
    """

    def __init__(self, expiry_minutes: int = 30):
        self._store: dict[str, MetadataCacheEntry] = {}
        self.expiry_minutes = expiry_minutes

    def _is_expired(self, entry: MetadataCacheEntry) -> bool:
        """Check if a cache entry has expired due to inactivity."""
        return datetime.now() - entry.update_timestamp > timedelta(
            minutes=self.expiry_minutes
        )

    def _cleanup_if_needed(self) -> None:
        """Lazy cleanup - remove all expired entries."""
        expired = [id for id, entry in self._store.items() if self._is_expired(entry)]
        for id in expired:
            del self._store[id]

    def _update_timestamp(self, cache_id: str) -> None:
        """Update the timestamp on an entry to prevent expiry."""
        if entry := self._store.get(cache_id):
            entry.update_timestamp = datetime.now()

    def cache_source(self, metadata: Source) -> str:
        """Cache source metadata and return ID."""
        self._cleanup_if_needed()
        cache_id = str(uuid.uuid4())

        self._store[cache_id] = MetadataCacheEntry(
            metadata=metadata,
            upload_schema=MetadataSchema.index,
            upload_type=BackendUploadType.INDEX,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=cache_id, status="awaiting_upload", entity=BackendUploadType.INDEX
            ),
        )
        return cache_id

    def cache_model(self, metadata: object) -> str:
        """Cache model results metadata and return ID."""
        raise NotImplementedError

    def get(self, cache_id: str) -> MetadataCacheEntry | None:
        """Retrieve metadata by ID if not expired. Updates timestamp on access."""
        self._cleanup_if_needed()

        entry = self._store.get(cache_id)
        if not entry:
            return None

        if self._is_expired(entry):
            del self._store[cache_id]
            return None

        self._update_timestamp(cache_id)
        return entry

    def update_status(
        self, cache_id: str, status: str, details: str | None = None
    ) -> bool:
        """Update the status of an entry. Returns False if entry not found."""
        if entry := self._store.get(cache_id):
            entry.status.status = status
            if details is not None:
                entry.status.details = details
            entry.update_timestamp = datetime.now()
            return True
        return False

    def remove(self, cache_id: str) -> bool:
        """Remove an entry from the store."""
        self._cleanup_if_needed()
        if cache_id in self._store:
            del self._store[cache_id]
            return True
        return False


async def process_upload(
    backend: MatchboxDBAdapter,
    upload_id: str,
    bucket: str,
    key: str,
    metadata_store: MetadataStore,
) -> None:
    """Background task to process uploaded file."""
    client = backend.settings.datastore.get_client()

    try:
        data_hashes = pa.Table.from_batches(
            [
                batch
                async for batch in s3_to_recordbatch(
                    client=client, bucket=bucket, key=key
                )
            ]
        )

        metadata_store.update_status(upload_id, "processing")

        backend.index(
            source=metadata_store.get(upload_id).metadata, data_hashes=data_hashes
        )

        metadata_store.update_status(upload_id, "complete")

    except Exception as e:
        metadata_store.update_status(upload_id, "failed", details=str(e))
