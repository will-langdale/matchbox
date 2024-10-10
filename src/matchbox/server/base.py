from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Iterable, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy.engine.result import ChunkedIteratorResult


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
    sha1: bytes
    left: bytes
    right: bytes
    probability: float = Field(default=None, ge=0, le=1)


class Cluster(BaseModel):
    parent: bytes
    child: bytes


class MatchboxModelAdapter(ABC):
    sha1: bytes
    name: str
    clusters: Countable
    probabilities: Countable

    @abstractmethod
    def insert_probabilities(self, probabilities: Iterable[Probability]) -> None: ...

    @abstractmethod
    def insert_clusters(self, clusters: Iterable[Cluster]) -> None: ...


class MatchboxDBAdapter(ABC):
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
    def index(self, db_schema: str, db_table: str) -> None: ...

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
