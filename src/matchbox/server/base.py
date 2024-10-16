from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd
from pydantic import AnyUrl, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rustworkx import PyDiGraph
from sqlalchemy import create_engine
from sqlalchemy import text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.exc import SQLAlchemyError


class Countable(Protocol):
    """A protocol for objects that can be counted."""

    def count(self) -> int: ...


class Listable(Protocol):
    """A protocol for objects that can be listed."""

    def list(self) -> list[str]: ...


class ListableAndCountable(Countable, Listable):
    """A protocol for objects that can be counted and listed."""

    pass


class Probability(BaseModel):
    """A probability of a match in the Matchbox database."""

    sha1: bytes
    left: bytes
    right: bytes
    probability: float = Field(default=None, ge=0, le=1)


class Cluster(BaseModel):
    """A cluster of data in the Matchbox database."""

    parent: bytes
    child: bytes


class SourceWarehouse(BaseModel):
    """A warehouse where source data for datasets in Matchbox can be found."""

    alias: str
    db_type: str
    username: str
    password: str = Field(repr=False)
    host: AnyUrl
    port: int
    _engine: Engine | None = None

    class Config:
        populate_by_name = True
        extra = "forbid"
        arbitrary_types_allowed = True

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connection_string = f"{self.db_type}://{self.username}:{self.password}@{self.host}:{self.port}"
            self._engine = create_engine(connection_string)
            self.test_connection()
        return self._engine

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(sqltext("SELECT 1"))
        except SQLAlchemyError:
            self._engine = None
            raise

    def __str__(self):
        return (
            f"SourceWarehouse(alias={self.alias}, type={self.db_type}, "
            f"host={self.host}, port={self.port})"
        )


class IndexableDataset(BaseModel):
    """A dataset that can be indexed in the Matchbox database."""

    database: SourceWarehouse
    db_pk: str
    db_schema: str
    db_table: str

    class Config:
        populate_by_name = True

    def __str__(self) -> str:
        return f"{self.db_schema}.{self.db_table}"


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
    ) -> pd.DataFrame | ChunkedIteratorResult: ...

    @abstractmethod
    def index(self, dataset: IndexableDataset) -> None: ...

    @abstractmethod
    def validate_hashes(
        self, hashes: list[bytes], hash_type: Literal["data", "cluster"]
    ) -> bool: ...

    @abstractmethod
    def get_model_subgraph(self) -> PyDiGraph: ...

    @abstractmethod
    def get_model(self, model: str) -> MatchboxModelAdapter: ...

    @abstractmethod
    def delete_model(self, model: str) -> None: ...

    @abstractmethod
    def insert_model(self, model: str) -> None: ...
