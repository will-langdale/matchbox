from typing import Dict, List, Literal, Union

import pandas as pd
from sqlalchemy import (
    Engine,
    MetaData,
    Table,
)
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from matchbox.server import MatchboxDBAdapter
from matchbox.server.exceptions import MatchboxSourceTableError, MatchboxValidatonError


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
        raise ValueError(
            f"""
            Could not identify schema and table in {full_name}.
        """
        )

    if validate and schema is None:
        raise MatchboxValidatonError(
            "Schema could not be detected and validation required."
        )

    return (schema, table)


def string_to_table(db_schema: str, db_table: str, engine: Engine) -> Table:
    """Takes strings and returns a SQLAlchemy Table."""
    with Session(engine) as session:
        source_schema = MetaData(schema=db_schema)
        try:
            source_table = Table(
                db_table,
                source_schema,
                schema=db_schema,
                autoload_with=session.get_bind(),
            )
        except NoSuchTableError as e:
            raise MatchboxSourceTableError(table_name=f"{db_schema}.{db_table}") from e

    return source_table


def schema_table_to_table(
    full_name: str, engine: Engine, validate: bool = False
) -> Table:
    """Thin wrapper combining get_schema_table_names and string_to_table."""

    db_schema, db_table = get_schema_table_names(full_name=full_name, validate=validate)
    source_table = string_to_table(
        db_schema=db_schema, db_table=db_table, engine=engine
    )

    return source_table


def selector(table: str, fields: list[str], engine: Engine) -> dict[str, list[str]]:
    """
    Takes the full name of a table and the fields you want to select from it,
    and arranges them in a dictionary parsable by query().

    Args:
        table: a table name in the form "schema.table".
        fields: a list of columns in the table
        engine: (optional) the engine to use to connect to
            your data warehouse

    Returns:
        A dictionary of the validated table name and fields
    """
    db_schema, db_table = get_schema_table_names(table, validate=True)
    selected_table = string_to_table(
        db_schema=db_schema, db_table=db_table, engine=engine
    )

    all_cols = set(selected_table.c.keys())
    selected_cols = set(fields)
    if not selected_cols <= all_cols:
        raise ValueError(
            f"{selected_cols.difference(all_cols)} not found in "
            f"{selected_table.schema}.{selected_table.name}"
        )

    return {f"{selected_table.schema}.{selected_table.name}": fields}


def selectors(*selector: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Builds individual selector dictionaries into one object appropriate for
    the query() function.

    Args:
        selector: any number of selectors

    Returns:
        A dictionary of selectors
    """
    return {k: v for d in (selector) for k, v in d.items()}


def query(
    selector: Dict[str, List[str]],
    backend: MatchboxDBAdapter,
    return_type: Literal["pandas", "sqlalchemy"],
    model: str | None = None,
    limit: int | None = None,
) -> Union[pd.DataFrame, ChunkedIteratorResult]:
    """Runs queries against the selected backend."""
    return backend.query(
        selector=selector,
        model=model,
        return_type=return_type,
        limit=limit,
    )
