"""Common database utilities for Matchbox."""

from typing import Any, Iterator, Literal, TypeVar, get_args, overload

import polars as pl
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.selectable import Select

ReturnTypeStr = Literal["arrow", "pandas", "polars"]
QueryReturnType = ArrowTable | PandasDataFrame | PolarsDataFrame

T = TypeVar("T")


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    iter_batches: Literal[False] = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    iter_batches: Literal[True],
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[QueryReturnType]: ...


def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: ReturnTypeStr = "pandas",
    *,
    iter_batches: bool = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Executes the given SQLAlchemy statement using Polars.

    Args:
        stmt (Select): A SQLAlchemy Select statement to be executed.
        engine (Engine): A SQLAlchemy Engine object for the database connection.
        return_type (str): The type of the return value. One of "arrow", "pandas",
            or "polars".
        iter_batches (bool): If True, return an iterator that yields each batch
            separately. If False, return a single DataFrame with all results.
            Default is False.
        batch_size (int | None): Indicate the size of each batch when processing
            data in batches. Default is None.
        schema_overrides (dict[str, Any] | None): A dictionary mapping column names
            to dtypes. Default is None.
        execute_options (dict[str, Any] | None): These options will be passed through
            into the underlying query execution method as kwargs. Default is None.

    Returns:
        If iter_batches is False: A dataframe of the query results in the specified
            format.
        If iter_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError: If the engine URL is not properly configured or if an unsupported
            return type is specified.
    """
    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    # Compile the SQLAlchemy statement to SQL string
    compiled_stmt = stmt.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    sql_query = str(compiled_stmt)

    # Extract the connection URL from the engine
    url: str | URL = engine.url
    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)
    if not isinstance(url, str):
        raise ValueError("Unable to obtain a valid connection string from the engine.")

    if iter_batches:
        results = pl.read_database(
            query=sql_query,
            connection=engine,
            iter_batches=True,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

        match return_type:
            case "polars":
                return results
            case "arrow":
                return (batch.to_arrow() for batch in results)
            case "pandas":
                return (batch.to_pandas() for batch in results)

    results = pl.read_database_uri(
        query=sql_query,
        uri=url,
        schema_overrides=schema_overrides,
        execute_options=execute_options,
    )

    match return_type:
        case "polars":
            return results
        case "arrow":
            return results.to_arrow()
        case "pandas":
            return results.to_pandas()


def get_schema_table_names(full_name: str) -> tuple[str, str]:
    """Takes a string table name and returns the unquoted schema and table as a tuple.

    Args:
        full_name: A string indicating a table's full name

    Returns:
        (schema, table): A tuple of schema and table name. If schema
            cannot be inferred, returns None.

    Raises:
        ValueError: When the function can't detect either a
            schema.table or table format in the input
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

    return schema, table


def fullname_to_prefix(fullname: str) -> str:
    """Converts a full name to a prefix for column names."""
    db_schema, db_table = get_schema_table_names(fullname)
    if db_schema:
        return f"{db_schema}_{db_table}_"
    return f"{db_table}_"
