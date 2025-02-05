import uuid
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict

from matchbox.common.arrow import SCHEMA_SOURCE_HASHES
from matchbox.common.sources import Source


class MetadataSchema(Enum):
    source = SCHEMA_SOURCE_HASHES


class MetadataCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: Source
    upload_schema: MetadataSchema
    timestamp: datetime


class MetadataStore:
    """A simple in-memory cache of uploaded metadata.

    Uses a janitor pattern for lazy cleanup.
    """

    def __init__(self, expiry_minutes: int = 30):
        self._store: dict[str, MetadataCacheEntry] = {}
        self.expiry_minutes = expiry_minutes

    def _is_expired(self, entry: MetadataCacheEntry) -> bool:
        """Check if a cache entry has expired."""
        return datetime.now() - entry.timestamp > timedelta(minutes=self.expiry_minutes)

    def _cleanup_if_needed(self) -> None:
        """Lazy cleanup - remove all expired entries."""
        expired = [id for id, entry in self._store.items() if self._is_expired(entry)]
        for id in expired:
            del self._store[id]

    def cache_source(self, metadata: Source) -> str:
        """Cache source metadata and return ID."""
        self._cleanup_if_needed()
        cache_id = str(uuid.uuid4())
        self._store[cache_id] = MetadataCacheEntry(
            metadata=metadata,
            upload_schema=MetadataSchema.source,
            timestamp=datetime.now(),
        )
        return cache_id

    def cache_model(self, metadata: object) -> str:
        """Cache model results metadata and return ID."""
        raise NotImplementedError

    def get(self, cache_id: str) -> MetadataCacheEntry | None:
        """Retrieve metadata by ID if not expired."""
        self._cleanup_if_needed()

        entry = self._store.get(cache_id)
        if not entry:
            return None

        if self._is_expired(entry):
            del self._store[cache_id]
            return None

        return entry

    def remove(self, cache_id: str) -> bool:
        """Remove an entry from the store."""
        self._cleanup_if_needed()
        if cache_id in self._store:
            del self._store[cache_id]
            return True
        return False
