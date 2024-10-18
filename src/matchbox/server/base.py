from abc import ABC, abstractmethod
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Protocol

from pandas import DataFrame
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rustworkx import PyDiGraph
from sqlalchemy import Engine
from sqlalchemy.engine.result import ChunkedIteratorResult

from matchbox.server.models import Cluster, Probability, Source


class Countable(Protocol):
    """A protocol for objects that can be counted."""

    def count(self) -> int: ...


class Listable(Protocol):
    """A protocol for objects that can be listed."""

    def list(self) -> list[str]: ...


class ListableAndCountable(Countable, Listable):
    """A protocol for objects that can be counted and listed."""

    pass


class MatchboxModelAdapter(ABC):
    """An abstract base class for Matchbox model adapters."""

    sha1: bytes
    name: str
    clusters: Countable
    probabilities: Countable

    @abstractmethod
    def insert_probabilities(
        self,
        probabilities: list[Probability],
        probability_type: Literal["deduplications", "links"],
        batch_size: int,
    ) -> None: ...

    @abstractmethod
    def insert_clusters(self, clusters: list[Cluster], batch_size: int) -> None: ...


class MatchboxBackends(StrEnum):
    """The available backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxSettings(BaseSettings):
    """Settings for the Matchbox application."""

    model_config = SettingsConfigDict(
        env_prefix="MB__",
        env_nested_delimiter="__",
        use_enum_values=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    datasets_config: Path
    batch_size: int = Field(default=250_000)
    backend_type: MatchboxBackends


class MatchboxDBAdapter(ABC):
    """An abstract base class for Matchbox database adapters."""

    settings: "MatchboxSettings"

    datasets: ListableAndCountable
    models: Countable
    models_from: Countable
    data: Countable
    clusters: Countable
    creates: Countable
    dedupes: Countable
    links: Countable
    proposes: Countable

    @abstractmethod
    def query(
        self,
        selector: dict[str, list[str]],
        model: str | None = None,
        return_type: Literal["pandas", "sqlalchemy"] | None = None,
        limit: int = None,
    ) -> DataFrame | ChunkedIteratorResult: ...

    @abstractmethod
    def index(self, dataset: Source) -> None: ...

    @abstractmethod
    def validate_hashes(
        self, hashes: list[bytes], hash_type: Literal["data", "cluster"]
    ) -> bool: ...

    @abstractmethod
    def get_dataset(self, db_schema: str, db_table: str, engine: Engine) -> Source: ...

    @abstractmethod
    def get_model_subgraph(self) -> PyDiGraph: ...

    @abstractmethod
    def get_model(self, model: str) -> MatchboxModelAdapter: ...

    @abstractmethod
    def delete_model(self, model: str, certain: bool) -> None: ...

    @abstractmethod
    def insert_model(self, model: str) -> None: ...

    @abstractmethod
    def clear(self, certain: bool) -> None: ...


class BackendManager:
    """Manages the Matchbox backend instance and settings."""

    _instance = None
    _settings = None

    @classmethod
    def initialise(cls, settings: "MatchboxSettings"):
        cls._settings = settings

    @classmethod
    def get_backend(cls) -> "MatchboxDBAdapter":
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")

        if cls._instance is None:
            BackendClass = get_backend_class(cls._settings.backend_type)
            cls._instance = BackendClass(cls._settings)
        return cls._instance

    @classmethod
    def get_settings(cls) -> "MatchboxSettings":
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")
        return cls._settings


def get_backend_settings(backend_type: MatchboxBackends) -> type[MatchboxSettings]:
    """Get the appropriate settings class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgresSettings

        return MatchboxPostgresSettings
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_backend_class(backend_type: MatchboxBackends) -> type[MatchboxDBAdapter]:
    """Get the appropriate backend class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgres

        return MatchboxPostgres
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def initialise_backend(settings: MatchboxSettings) -> None:
    """Utility function to initialise the Matchbox backend based on settings."""
    BackendManager.initialise(settings)


def initialise_matchbox() -> None:
    """Initialise the Matchbox backend based on environment variables."""
    base_settings = MatchboxSettings()

    SettingsClass = get_backend_settings(base_settings.backend_type)
    settings = SettingsClass()

    initialise_backend(settings)


def inject_backend(func: Callable) -> Callable:
    """Decorator to inject the Matchbox backend into functions.

    Used to allow user-facing functions to access the backend without needing to
    pass it in. The backend is defined by the MB__BACKEND_TYPE environment variable.

    If the user specifies a backend, it will be used instead of the injection.
    """

    @wraps(func)
    def _inject_backend(*args, backend: "MatchboxDBAdapter | None" = None, **kwargs):
        if backend is None:
            backend = BackendManager.get_backend()
        return func(backend, *args, **kwargs)

    return _inject_backend
