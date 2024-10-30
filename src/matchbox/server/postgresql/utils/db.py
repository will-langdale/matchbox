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
from sqlalchemy.orm import DeclarativeMeta, Session

from matchbox.server.postgresql.orm import Models, ModelsFrom, ModelType, Sources

# Retrieval


def get_model_subgraph(engine: Engine) -> rx.PyDiGraph:
    """Retrieves the model subgraph as a PyDiGraph."""
    G = rx.PyDiGraph()
    models = {}
    datasets = {}

    with Session(engine) as session:
        for dataset in session.query(Sources).all():
            dataset_idx = G.add_node(
                {
                    "id": str(dataset.model),
                    "name": f"{dataset.schema}.{dataset.table}",
                    "type": "dataset",
                }
            )
            datasets[dataset.model] = dataset_idx

        for model in session.query(Models).all():
            model_idx = G.add_node(
                {"id": str(model.hash), "name": model.name, "type": "model"}
            )
            models[model.hash] = model_idx
            if model.type == ModelType.DATASET:
                dataset_idx = datasets.get(model.hash)
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
    records: list[tuple], table: Table, batch_size: int
) -> Callable[[str], Tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def _batches(
        high_watermark,  # noqa ARG001 required for pg_bulk_ingest
    ) -> Iterable[Tuple[None, None, Iterable[Tuple[Table, tuple]]]]:
        for batch in batched(records, batch_size):
            yield None, None, ((table, t) for t in batch)

    return _batches


def batch_ingest(
    records: list[tuple[Any]],
    table: DeclarativeMeta,
    conn: Connection,
    batch_size: int,
) -> None:
    """Batch ingest records into a database table.

    We isolate the table and metadata as pg_bulk_ingest will try and drop unrelated
    tables if they're in the same schema.
    """

    isolated_metadata = MetaData(schema=table.__table__.schema)
    isolated_table = Table(
        table.__table__.name,
        isolated_metadata,
        *[c.copy() for c in table.__table__.columns],
        schema=table.__table__.schema,
    )

    fn_batch = data_to_batch(
        records=records,
        table=isolated_table,
        batch_size=batch_size,
    )

    ingest(
        conn=conn,
        metadata=isolated_metadata,
        batches=fn_batch,
        upsert=Upsert.IF_PRIMARY_KEY,
        delete=Delete.OFF,
    )
