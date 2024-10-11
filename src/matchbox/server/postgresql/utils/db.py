import contextlib
import cProfile
import io
import pstats
from itertools import islice
from typing import Any, Callable, Iterable, Tuple

import rustworkx as rx
from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.engine.base import Connection
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.server.postgresql import Models, ModelsFrom, SourceDataset

# Data conversion


def dataset_to_table(dataset: SourceDataset, engine: Engine) -> Table:
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
            raise MatchboxSourceTableError(
                table_name=f"{dataset.db_schema}.{dataset.db_table}"
            ) from e

    return source_table


def string_to_dataset(db_schema: str, db_table: str, engine: Engine) -> SourceDataset:
    """Takes strings and returns a CMF SourceDataset"""
    with Session(engine) as session:
        dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=db_schema, db_table=db_table)
            .first()
        )
    return dataset


# Retrieval


def get_model_subgraph(engine: Engine) -> rx.PyDiGraph:
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
    records: list[dict], table: Table, batch_size: int
) -> Callable[[str], Tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def _batches() -> Iterable[Tuple[None, None, Iterable[Tuple[Table, dict]]]]:
        for batch in batched(records, batch_size):
            yield None, None, ((table, t) for t in batch)

    return _batches


def batch_ingest(
    records: list[dict],
    table: Table,
    conn: Connection,
    batch_size: int,
) -> None:
    """Batch ingest records into a database table."""

    fn_batch = data_to_batch(
        records=records,
        table=table,
        batch_size=batch_size,
    )

    ingest(
        conn=conn,
        metadata=table.metadata,
        batches=fn_batch,
        upsert=Upsert.IF_PRIMARY_KEY,
        delete=Delete.OFF,
    )
