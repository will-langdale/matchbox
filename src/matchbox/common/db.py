"""Common database utilities for Matchbox."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    TypeVar,
    get_args,
    overload,
)

import polars as pl
import sqlglot
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from pydantic import AnyUrl
from sqlalchemy.engine import Engine

from matchbox.common.exceptions import MatchboxSourceExtractTransformError

if TYPE_CHECKING:
    from adbc_driver_postgresql.dbapi import Connection as ADBCConnection
else:
    ADBCConnection = Any

ReturnTypeStr = Literal["arrow", "pandas", "polars"]
QueryReturnType = ArrowTable | PandasDataFrame | PolarsDataFrame

T = TypeVar("T")


@overload
def sql_to_df(
    stmt: str,
    connection: Engine | ADBCConnection,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType: ...


@overload
def sql_to_df(
    stmt: str,
    connection: Engine | ADBCConnection,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[QueryReturnType]: ...


def sql_to_df(
    stmt: str,
    connection: Engine | ADBCConnection,
    return_type: ReturnTypeStr = "pandas",
    *,
    return_batches: bool = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Executes the given SQLAlchemy statement or SQL string using Polars.

    Args:
        stmt (str): A SQL string to be executed.
        connection (Engine | ADBCConnection): A SQLAlchemy Engine object or
            ADBC connection.
        return_type (str): The type of the return value. One of "arrow", "pandas",
            or "polars".
        return_batches (bool): If True, return an iterator that yields each batch
            separately. If False, return a single DataFrame with all results.
            Default is False.
        batch_size (int | None): Indicate the size of each batch when processing
            data in batches. Default is None.
        rename (dict[str, str] | Callable | None): A dictionary mapping old column
            names to new column names, or a callable that takes a DataFrame and
            returns a DataFrame with renamed columns. Default is None.
        schema_overrides (dict[str, Any] | None): A dictionary mapping column names
            to dtypes. Default is None.
        execute_options (dict[str, Any] | None): These options will be passed through
            into the underlying query execution method as kwargs. Default is None.

    Returns:
        If return_batches is False: A dataframe of the query results in the specified
            format.
        If return_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError:

            * If the connection is not properly configured or if an unsupported
                return type is specified.
            * If batch_size and return_batches are either both set or both unset.

    """
    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    if not batch_size and return_batches:
        raise ValueError("A batch size must be specified if return_batches is True")

    if batch_size and not return_batches:
        raise ValueError("Cannot set a batch size if return_batches if False")

    def _to_format(results: PolarsDataFrame) -> QueryReturnType:
        """Convert the results to the specified format."""
        if rename:
            results = results.rename(rename)

        if return_type == "polars":
            return results
        elif return_type == "arrow":
            return results.to_arrow()
        elif return_type == "pandas":
            return results.to_pandas()

    res = pl.read_database(
        query=stmt,
        connection=connection,
        iter_batches=bool(batch_size),
        batch_size=batch_size,
        schema_overrides=schema_overrides,
        execute_options=execute_options,
    )

    if return_batches:
        return (_to_format(batch) for batch in res)

    return _to_format(res)


def validate_sql_for_data_extraction(sql: str) -> bool:
    """Validates that the SQL statement only contains a single data-extracting command.

    Args:
        sql: The SQL statement to validate

    Returns:
        bool: True if the SQL statement is valid

    Raises:
        ParseError: If the SQL statement cannot be parsed
        MatchboxSourceExtractTransformError: If validation requirements are not met
    """
    if not sql.strip():
        raise MatchboxSourceExtractTransformError(
            "SQL statement is empty or only contains whitespace."
        )

    expressions = sqlglot.parse(sql)

    if len(expressions) > 1:
        raise MatchboxSourceExtractTransformError(
            "SQL statement contains multiple commands."
        )

    if not expressions:
        raise MatchboxSourceExtractTransformError(
            "SQL statement does not contain any valid expressions."
        )

    expr = expressions[0]

    if not isinstance(expr, sqlglot.expressions.Select):
        raise MatchboxSourceExtractTransformError(
            "SQL statement must start with a SELECT or WITH command."
        )

    forbidden = (
        sqlglot.expressions.DDL,
        sqlglot.expressions.DML,
        sqlglot.expressions.Into,
    )

    if len(list(expr.find_all(forbidden))) > 0:
        raise MatchboxSourceExtractTransformError(
            "SQL statement must not contain DDL or DML commands."
        )

    return True


def clean_uri(
    uri: str | AnyUrl,
    strip_driver: bool = True,
    strip_credentials: bool = True,
    strip_query_fragment: bool = True,
) -> AnyUrl:
    """Clean a database URI.

    Optionally removes driver, credentials, and/or query/fragment components.

    Args:
        uri: A database URI as a string or AnyUrl object
        strip_driver: Whether to strip the driver component
            (e.g., 'postgresql+psycopg2' -> 'postgresql')
        strip_credentials: Whether to strip username and password components
        strip_query_fragment: Whether to strip query and fragment components

    Returns:
        An AnyUrl object with the specified components removed
    """
    if isinstance(uri, str):
        uri = AnyUrl(uri)

    # Strip driver if requested
    scheme = uri.scheme
    if strip_driver and "+" in scheme:
        scheme = scheme.split("+")[0]

    # Build netloc (username:password@host:port)
    netloc = ""
    if not strip_credentials and (uri.username or uri.password):
        auth = uri.username or ""
        if uri.password:
            auth += f":{uri.password}"
        netloc += f"{auth}@"

    if uri.host:
        netloc += uri.host
        if uri.port:
            netloc += f":{uri.port}"

    # Build path
    path = uri.path or ""

    # Build query and fragment
    query = "" if strip_query_fragment else (f"?{uri.query}" if uri.query else "")
    fragment = (
        "" if strip_query_fragment else (f"#{uri.fragment}" if uri.fragment else "")
    )

    # Construct the URI
    new_uri = f"{scheme}://{netloc}{path}{query}{fragment}"

    return AnyUrl(new_uri)
