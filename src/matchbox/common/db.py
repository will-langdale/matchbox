from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload

import connectorx as cx
import pyarrow as pa
from matchbox.common.exceptions import MatchboxValidatonError
from matchbox.common.hash import HASH_FUNC
from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    Engine,
    MetaData,
    Select,
    Table,
    create_engine,
    select,
)
from sqlalchemy import text as sqltext
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

ReturnTypeStr = Literal["arrow", "pandas", "polars"]

T = TypeVar("T")


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
        metadata = MetaData(schema=self.db_schema)
        table = Table(self.db_table, metadata, autoload_with=self.database.engine)
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
        table = self.to_table()
        schema_representation = f"{str(self)}: " + ",".join(
            f"{col.name}:{str(col.type)}" for col in table.columns
        )
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
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="pandas")


def convert_large_binary_to_binary(table: pa.Table) -> pa.Table:
    """Converts Arrow large_binary fields to binary type."""
    new_fields = []
    for field in table.schema:
        if pa.types.is_large_binary(field.type):
            new_fields.append(field.with_type(pa.binary()))
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["arrow"]
) -> pa.Table: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["pandas"]
) -> DataFrame: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["polars"]
) -> PolarsDataFrame: ...


def sql_to_df(
    stmt: Select, engine: Engine, return_type: ReturnTypeStr = "pandas"
) -> pa.Table | DataFrame | PolarsDataFrame:
    """
    Executes the given SQLAlchemy statement using connectorx.

    Args:
        stmt (Select): A SQLAlchemy Select statement to be executed.
        engine (Engine): A SQLAlchemy Engine object for the database connection.

    Returns:
        A dataframe of the query results.

    Raises:
        ValueError: If the engine URL is not properly configured.
    """
    compiled_stmt = stmt.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    sql_query = str(compiled_stmt)

    url: Union[str, URL] = engine.url

    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)

    if not isinstance(url, str):
        raise ValueError("Unable to obtain a valid connection string from the engine.")

    result = cx.read_sql(conn=url, query=sql_query, return_type=return_type)

    if return_type == "arrow":
        return convert_large_binary_to_binary(table=result)

    return result


def get_schema_table_names(full_name: str, validate: bool = False) -> tuple[str, str]:
    """
    Takes a string table name and returns the unquoted schema and
    table as a tuple. If you insert these into a query, you need to
    add double quotes in from statements, or single quotes in where.

    Parameters:
        full_name: A string indicating a Postgres table
        validate: Whether to error if both schema and table aren't
        detected

    Raises:
        ValueError: When the function can't detect either a
        schema.table or table format in the input
        MatchboxValidatonError: If both schema and table can't be detected
        when the validate argument is True

    Returns:
        (schema, table): A tuple of schema and table name. If schema
        cannot be inferred, returns None.
    """

    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(f"Could not identify schema and table in {full_name}.")

    if validate and schema is None:
        raise MatchboxValidatonError(
            "Schema could not be detected and validation required."
        )

    return (schema, table)
