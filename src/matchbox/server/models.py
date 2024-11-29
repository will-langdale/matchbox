from typing import TypeVar

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    MetaData,
    Table,
    create_engine,
    select,
)
from sqlalchemy import text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.hash import HASH_FUNC

T = TypeVar("T")

# TEMPORARY
import logging

logger = logging.getLogger("cmf_pipelines")


class Probability(BaseModel):
    """A probability of a match in the Matchbox database.

    A probability describes the likelihood of a match between two clusters.
    """

    hash: bytes
    left: bytes
    right: bytes
    probability: float = Field(default=None, ge=0, le=1)


class Cluster(BaseModel):
    """A cluster of data in the Matchbox database.

    A cluster describes a single entity resolved at the specified probability
    threshold or higher.
    """

    parent: bytes
    children: set[bytes]
    threshold: float = Field(default=None, ge=0, le=1)


class SourceWarehouse(BaseModel):
    """A warehouse where source data for datasets in Matchbox can be found."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    alias: str
    db_type: str
    user: str
    password: str = Field(repr=False)
    host: str
    port: int
    database: str
    _engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connection_string = f"{self.db_type}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
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
            f"host={self.host}, port={self.port}, database={self.database})"
        )

    @classmethod
    def from_engine(cls, engine: Engine, alias: str | None = None) -> "SourceWarehouse":
        """Create a SourceWarehouse instance from an SQLAlchemy Engine object."""
        url = engine.url

        warehouse = cls(
            alias=alias or url.database,
            db_type=url.drivername,
            user=url.username,
            password=url.password,
            host=url.host,
            port=url.port or 0,
            database=url.database,
        )
        _ = warehouse.engine

        return warehouse


class Source(BaseModel):
    """A dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    database: SourceWarehouse | None = None
    db_pk: str
    db_schema: str
    db_table: str

    def __str__(self) -> str:
        return f"{self.db_schema}.{self.db_table}"

    def __hash__(self) -> int:
        return hash(
            (type(self), self.db_pk, self.db_schema, self.db_table, self.database.alias)
        )

    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        logger.info("Matchbox: Start of to_table()")
        logger.info("Matchbox: MetaData()")
        metadata = MetaData(schema=self.db_schema)
        logger.info("Matchbox: Table()")
        table = Table(self.db_table, metadata, autoload_with=self.database.engine)
        logger.info("Matchbox: End of to_table()")
        return table

    def _select(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        if fields:
            stmt = select(table.c[tuple(fields)])
        else:
            stmt = select(table)

        if pks:
            stmt = stmt.where(table.c[self.db_pk].in_(pks))

        if limit:
            stmt = stmt.limit(limit)

        return stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    def to_hash(self) -> bytes:
        """Generate a unique hash based on the table's columns and datatypes."""
        logger.info("Matchbox: Start of to_hash()")
        logger.info("Matchbox: to_table()")
        table = self.to_table()

        # Original
        # schema_representation = ",".join(
        #     f"{col.name}:{str(col.type)}" for col in table.columns
        # )

        schema_representation = f"{str(self)}: " + ",".join(
            f"{col.name}:{str(col.type)}" for col in table.columns
        )
        logger.info("Matchbox: End of to_hash()")
        return HASH_FUNC(schema_representation.encode("utf-8")).digest()

    def to_arrow(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        """Returns the dataset as a PyArrow Table."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="arrow")

    def to_pandas(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> DataFrame:
        """Returns the dataset as a pandas DataFrame."""
        logger.info("Matchbox: Start of to_pandas()")
        logger.info("Matchbox: _select()")
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        logger.info("Matchbox: sql_to_df")
        return sql_to_df(stmt, self.database.engine, return_type="pandas")
