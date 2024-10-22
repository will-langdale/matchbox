from typing import Literal

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy import Engine, inspect

from matchbox.common.db import get_schema_table_names
from matchbox.server import MatchboxDBAdapter, inject_backend
from matchbox.server.models import Source


@inject_backend
def selector(
    backend: MatchboxDBAdapter, table: str, fields: list[str], engine: Engine
) -> dict[Source, list[str]]:
    """
    Takes the full name of a table and the fields you want to select from it,
    and arranges them in a dictionary parsable by query().

    Args:
        table: a table name in the form "schema.table".
        fields: a list of columns in the table
        engine: the engine to use to connect to
            your data warehouse

    Returns:
        A dictionary of the validated Source and fields
    """
    db_schema, db_table = get_schema_table_names(table, validate=True)
    dataset = backend.get_dataset(db_schema=db_schema, db_table=db_table, engine=engine)

    # Validate the fields
    inspector = inspect(engine)
    all_cols = set(
        column["name"]
        for column in inspector.get_columns(dataset.db_table, schema=dataset.db_schema)
    )
    selected_cols = set(fields)
    if not selected_cols <= all_cols:
        raise ValueError(f"{selected_cols.difference(all_cols)} not found in {dataset}")

    return {dataset: fields}


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


@inject_backend
def query(
    backend: MatchboxDBAdapter,
    selector: dict[str, list[str]],
    return_type: Literal["pandas", "arrow"],
    model: str | None = None,
    limit: int | None = None,
) -> DataFrame | ArrowTable:
    """Runs queries against the selected backend."""
    return backend.query(
        selector=selector,
        model=model,
        return_type=return_type,
        limit=limit,
    )
