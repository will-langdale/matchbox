from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Iterable, Literal, Protocol

import pandas as pd
from pydantic import AnyUrl, BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy import text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.exc import SQLAlchemyError

from matchbox.server.postgresql.adapter import MatchboxPostgres


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
        allow_population_by_field_name = True
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
        allow_population_by_field_name = True

    def __str__(self) -> str:
        return f"{self.db_schema}.{self.db_table}"


class MatchboxModelAdapter(ABC):
    """An abstract base class for Matchbox model adapters."""

    sha1: bytes
    name: str
    clusters: Countable
    probabilities: Countable

    @abstractmethod
    def insert_probabilities(self, probabilities: Iterable[Probability]) -> None: ...

    @abstractmethod
    def insert_clusters(self, clusters: Iterable[Cluster]) -> None: ...


class MatchboxDBAdapter(ABC):
    """An abstract base class for Matchbox database adapters."""

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
    def get_model_subgraph(self) -> dict: ...

    @abstractmethod
    def get_model(self, model: str) -> MatchboxModelAdapter: ...

    @abstractmethod
    def delete_model(self, model: str) -> None: ...

    @abstractmethod
    def insert_model(self, model: str) -> None: ...


class MatchboxBackends(StrEnum):
    """The available backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxSettings(BaseModel):
    """Settings for the Matchbox application."""

    batch_size: int = Field(default=250_000, alias="MB__BATCH_SIZE")
    backend_type: MatchboxBackends = Field(
        default=MatchboxBackends.POSTGRES, alias="MB__BACKEND_TYPE"
    )

    class Config:
        env_prefix = "MB__"
        use_enum_values = True
        allow_population_by_field_name = True

    @property
    def backend(self) -> MatchboxDBAdapter:
        if self.backend_type == MatchboxBackends.POSTGRES:
            return MatchboxPostgres(settings=self)
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")
