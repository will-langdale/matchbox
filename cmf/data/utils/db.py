import contextlib
import cProfile
import io
import pstats
from itertools import islice
from typing import Any, Callable, Iterable, Tuple

import rustworkx as rx
from pandas import DataFrame
from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from cmf.data import ENGINE, Models, ModelsFrom, SourceDataset
from cmf.data.exceptions import CMFSourceTableError

# Data conversion


def get_schema_table_names(full_name: str, validate: bool = False) -> (str, str):
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
        ValidationError: If both schema and table can't be detected
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
        raise ("Schema could not be detected and validation required.")

    return (schema, table)


def dataset_to_table(dataset: SourceDataset, engine: Engine = ENGINE) -> Table:
    """Takes a CMF SourceDataset object and returns a SQLAlchemy Table."""
    with Session(engine) as session:
        source_schema = MetaData(schema=dataset.db_schema)
        try:
            source_table = Table(
                dataset.db_table,
                source_schema,
                schema=dataset.db_schema,
                autoload_with=session.get_bind(),
            )
        except NoSuchTableError as e:
            raise CMFSourceTableError(
                table_name=f"{dataset.db_schema}.{dataset.db_table}"
            ) from e

    return source_table


def string_to_table(db_schema: str, db_table: str, engine: Engine = ENGINE) -> Table:
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
            raise CMFSourceTableError(table_name=f"{db_schema}.{db_table}") from e

    return source_table


def schema_table_to_table(
    full_name: str, validate: bool = False, engine: Engine = ENGINE
) -> Table:
    """Thin wrapper combining get_schema_table_names and string_to_table."""

    db_schema, db_table = get_schema_table_names(full_name=full_name, validate=validate)
    source_table = string_to_table(
        db_schema=db_schema, db_table=db_table, engine=engine
    )

    return source_table


def string_to_dataset(
    db_schema: str, db_table: str, engine: Engine = ENGINE
) -> SourceDataset:
    """Takes strings and returns a CMF SourceDataset"""
    with Session(engine) as session:
        dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=db_schema, db_table=db_table)
            .first()
        )
    return dataset


# Retrieval


def get_model_subgraph(engine: Engine = ENGINE) -> rx.PyDiGraph:
    """Retrieves the model subgraph as a PyDiGraph."""
    G = rx.PyDiGraph()
    models = {}
    datasets = {}

    with Session(engine) as session:
        for dataset in session.query(SourceDataset).all():
            dataset_idx = G.add_node(
                {
                    "id": str(dataset.uuid),
                    "name": f"{dataset.db_schema}.{dataset.db_table}",
                    "type": "dataset",
                }
            )
            datasets[dataset.uuid] = dataset_idx

        for model in session.query(Models).all():
            model_idx = G.add_node(
                {"id": str(model.sha1), "name": model.name, "type": "model"}
            )
            models[model.sha1] = model_idx
            if model.deduplicates is not None:
                dataset_idx = datasets.get(model.deduplicates)
                _ = G.add_edge(model_idx, dataset_idx, {"type": "deduplicates"})

        for edge in session.query(ModelsFrom).all():
            parent_idx = models.get(edge.parent)
            child_idx = models.get(edge.child)
            _ = G.add_edge(parent_idx, child_idx, {"type": "from"})

    return G


# SQLAlchemy profiling


@contextlib.contextmanager
def sqa_profiled():
    """SQLAlchemy profiler.

    Taken directly from their docs:
    https://docs.sqlalchemy.org/en/20/faq/performance.html#query-profiling
    """
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())


# Misc


def batched(iterable: Iterable, n: int) -> Iterable:
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def data_to_batch(
    dataframe: DataFrame, table: Table, batch_size: int
) -> Callable[[str], Tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def batches(high_watermark):
        for records in batched(dataframe.to_records(index=None), batch_size):
            yield None, None, ((table, (t)) for t in records)

    return batches
