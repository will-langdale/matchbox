"""Base class for client-side DAG step nodes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import polars as pl
import pyarrow as pa

from matchbox.client import _handler
from matchbox.common.arrow import check_schema
from matchbox.common.dtos import (
    Resolution,
    ResolutionName,
    ResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger, profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
else:
    DAG = Any

T = TypeVar("T")


class StepConfigProtocol(Protocol):
    """Minimal protocol required by client DAG step config DTOs."""

    @property
    def dependencies(self) -> list[ResolutionName]:
        """Execution prerequisites required before running the step."""
        ...

    @property
    def parents(self) -> list[ResolutionName]:
        """Direct DAG edges to this step."""
        ...

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialise the config for stable hashing."""
        ...


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after step run.

    Raises:
        RuntimeError: If the step hasn't been run yet.
    """

    @wraps(method)
    def wrapper(self: "Step", *args: Any, **kwargs: Any) -> T:
        if self._local_data is None:
            raise RuntimeError("The step must be run before attempting this operation.")
        return method(self, *args, **kwargs)

    return wrapper


class Step(ABC):
    """Base class for client-side DAG nodes that compute and sync data."""

    _local_data_schema: ClassVar[pa.Schema]

    def __init__(
        self,
        dag: DAG,
        name: str,
        description: str | None = None,
    ) -> None:
        """Initialise the step."""
        self.dag = dag
        self.name = name
        self.description = description
        self._local_data: pl.DataFrame | None = None

    # Local data access

    @property
    def local_data(self) -> pl.DataFrame | None:
        """The locally computed results for this step."""
        return self._local_data

    def clear_data(self) -> None:
        """Drop locally computed data."""
        self._local_data = None

    # Abstract interface

    @property
    @abstractmethod
    def resolution_path(self) -> ResolutionPath:
        """The resolution path used to identify this step on the server."""
        ...

    @property
    @abstractmethod
    def sources(self) -> set[SourceResolutionName]:
        """Set of source names upstream of this node."""
        ...

    @property
    @abstractmethod
    def config(self) -> StepConfigProtocol:
        """Config DTO for this step."""
        ...

    @abstractmethod
    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        ...

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        resolution_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "Step":
        """Reconstruct from Resolution. Subclasses should override this."""
        raise NotImplementedError(f"{cls.__name__} must implement from_resolution.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Execute the step, populate _local_data, and return it."""
        ...

    # Concrete shared behaviour

    @property
    def cache_path(self) -> Path:
        """Path within the DAG cache for storing this step's local data."""
        return self.dag.cache_path / f"{self.name}.parquet"

    def __hash__(self) -> int:
        """Return a hash of the step based on its config."""
        return hash(self.config.model_dump_json())

    def __eq__(self, other: object) -> bool:
        """Check equality of two step objects based on their config."""
        if type(other) is not type(self):
            return False
        return self.config == other.config

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute a content hash of the local data for fingerprinting."""
        check_schema(
            expected=self._local_data_schema, actual=self._local_data.to_arrow().schema
        )
        return hash_arrow_table(self._local_data.to_arrow())

    def delete(self, certain: bool = False) -> bool:
        """Delete this step and its associated data from the backend."""
        return _handler.delete_resolution(
            path=self.resolution_path,
            certain=certain,
        ).success

    def download(self) -> pl.DataFrame:
        """Fetch remote data for this step and store it locally."""
        table = _handler.get_data(path=self.resolution_path)
        check_schema(expected=self._local_data_schema, actual=table.schema)
        self._local_data = pl.from_arrow(table)
        return self._local_data

    def prepare(self) -> None:
        """Ensure local execution prerequisites are available."""
        for dependency_name in self.config.dependencies:
            dependency = self.dag.nodes[dependency_name]
            if dependency.local_data is None:
                dependency.download()

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """Send step config and local data to the server.

        Not resistant to race conditions: only one client should call sync at a time.
        """
        log_prefix = f"Sync {self.name}"
        resolution = self.to_resolution()

        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                _handler.update_resolution(
                    resolution=resolution,
                    path=self.resolution_path,
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)
            logger.info("Setting data for new resolution", prefix=log_prefix)
            _handler.set_data(path=self.resolution_path, data=self._local_data)
