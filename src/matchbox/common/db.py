from typing import TYPE_CHECKING, Any, Literal, Union, overload

import connectorx as cx
import pyarrow as pa
from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.selectable import Select

from matchbox.common.exceptions import MatchboxValidatonError

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

ReturnTypeStr = Literal["arrow", "pandas", "polars"]


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: str
    source_id: set[str] = Field(default_factory=set)
    target: str
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self


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
