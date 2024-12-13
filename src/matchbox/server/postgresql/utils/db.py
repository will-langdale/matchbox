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

from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    ResolutionFrom,
    Resolutions,
    Sources,
)

# Retrieval


def get_resolution_graph(engine: Engine) -> ResolutionGraph:
    """Retrieves the resolution graph."""
    G = ResolutionGraph(nodes=set(), edges=set())
    with Session(engine) as session:
        for resolution in session.query(Resolutions).all():
            G.nodes.add(
                ResolutionNode(
                    hash=resolution.hash,
                    name=resolution.name,
                    type=ResolutionNodeType(resolution.type),
                )
            )

        for edge in (
            session.query(ResolutionFrom).filter(ResolutionFrom.level == 1).all()
        ):
            G.edges.add(ResolutionEdge(parent=edge.parent, child=edge.child))

    return G


def get_data_subgraph(engine: Engine) -> rx.PyDiGraph:
    """Retrieves the complete data subgraph as a PyDiGraph."""
    G = rx.PyDiGraph()
    nodes = {}

    with Session(engine) as session:
        sources = {source.model: source for source in session.query(Sources).all()}

        for source in sources.values():
            source_id = f"{source.schema}.{source.table}"
            if source_id not in nodes:
                source_idx = G.add_node({"id": source_id, "type": "source"})
                nodes[source_id] = source_idx

        for cluster in session.query(Clusters).all():
            cluster_id = cluster.hash
            if cluster_id not in nodes:
                cluster_idx = G.add_node({"id": cluster_id, "type": "cluster"})
                nodes[cluster_id] = cluster_idx

            if cluster.id is not None and cluster.dataset is not None:
                source = sources[cluster.dataset]
                data_id = str(cluster.id)
                data_idx = G.add_node({"id": data_id, "type": "data"})

                source_id = f"{source.schema}.{source.table}"
                G.add_edge(data_idx, nodes[source_id], {"type": "source"})
                G.add_edge(nodes[cluster_id], data_idx, {"type": "data"})

        for contains in session.query(Contains).all():
            G.add_edge(
                nodes[contains.parent], nodes[contains.child], {"type": "contains"}
            )

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
        *[c._copy() for c in table.__table__.columns],
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
