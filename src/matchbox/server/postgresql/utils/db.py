import contextlib
import cProfile
import io
import pstats
from datetime import datetime
from itertools import islice
from typing import Any, Callable, Iterable

import adbc_driver_postgresql.dbapi
import pyarrow as pa
from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, Index, MetaData, Table, func, text
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeMeta, Session

from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ResolutionFrom,
    Resolutions,
)

# Retrieval


def resolve_model_name(model: str, engine: Engine) -> Resolutions:
    """Resolves a model name to a Resolution object.

    Args:
        model: The name of the model to resolve.

    Raises:
        MatchboxResolutionNotFoundError: If the model doesn't exist.
    """
    with Session(engine) as session:
        if resolution := session.query(Resolutions).filter_by(name=model).first():
            return resolution
        raise MatchboxResolutionNotFoundError(resolution_name=model)


def get_resolution_graph(engine: Engine) -> ResolutionGraph:
    """Retrieves the resolution graph."""
    G = ResolutionGraph(nodes=set(), edges=set())
    with Session(engine) as session:
        for resolution in session.query(Resolutions).all():
            G.nodes.add(
                ResolutionNode(
                    id=resolution.resolution_id,
                    name=resolution.name,
                    type=ResolutionNodeType(resolution.type),
                )
            )

        for edge in (
            session.query(ResolutionFrom).filter(ResolutionFrom.level == 1).all()
        ):
            G.edges.add(ResolutionEdge(parent=edge.parent, child=edge.child))

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
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def data_to_batch(
    records: list[tuple], table: Table, batch_size: int
) -> Callable[[str], tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def _batches(
        high_watermark,  # noqa ARG001 required for pg_bulk_ingest
    ) -> Iterable[tuple[None, None, Iterable[tuple[Table, tuple]]]]:
        for batch in batched(records, batch_size):
            yield None, None, ((table, t) for t in batch)

    return _batches


def isolate_table(table: DeclarativeMeta) -> tuple[MetaData, Table]:
    """Creates an isolated copy of a SQLAlchemy table.

    This is used to prevent pg_bulk_ingest from attempting to drop unrelated tables
    in the same schema. The function creates a new Table instance with:

    * A fresh MetaData instance
    * Copied columns
    * Recreated indices properly bound to the new table

    Args:
        table: The DeclarativeMeta class whose table should be isolated

    Returns:
        A tuple of:
            * The isolated SQLAlchemy MetaData
            * A new SQLAlchemy Table instance with all columns and indices
    """
    isolated_metadata = MetaData(schema=table.__table__.schema)

    isolated_table = Table(
        table.__table__.name,
        isolated_metadata,
        *[c._copy() for c in table.__table__.columns],
        schema=table.__table__.schema,
    )

    for idx in table.__table__.indexes:
        Index(
            idx.name,
            *[isolated_table.c[col.name] for col in idx.columns],
            **{k: v for k, v in idx.kwargs.items()},
        )

    return isolated_metadata, isolated_table


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


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
    isolated_metadata, isolated_table = isolate_table(table=table)

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


# TODO: replace batch_ingest with large_ingest across the codebase
# TODO: allow custom subset selection before table copy
def large_ingest(data: pa.Table, table_class: DeclarativeMeta):
    """
    Saves a PyArrow Table to PostgreSQL using ADBC.
    """
    with (
        adbc_driver_postgresql.dbapi.connect(MBDB.connection_string) as conn,
        conn.cursor() as cursor,
        MBDB.get_session() as session,
    ):
        table: Table = table_class.__table__
        metadata = table.metadata

        try:
            # Create temp table
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            temp_table_name = f"{table.name}_tmp_{timestamp}"
            temp_table = Table(
                temp_table_name, metadata, *[c.copy() for c in table.columns]
            )
            temp_table.create(session.bind)

            # Copy old records to temp table
            insert_stmt = temp_table.insert().from_select(
                [c.name for c in table.columns], table.select()
            )
            session.execute(insert_stmt)
            session.commit()

            # Add new records
            batch_reader = pa.RecordBatchReader.from_batches(
                data.schema, data.to_batches()
            )
            cursor.adbc_ingest(
                table_name=temp_table_name,
                data=batch_reader,
                mode="append",
                db_schema_name=table.schema,
            )
            conn.commit()
            with session.begin():
                # TODO: need to deal with inbound foreign keys ahead of table dropping
                # Swap temp and original table
                table.drop(session.bind)
                session.execute(
                    text(
                        f"""ALTER TABLE {table.schema}.{temp_table_name}
                        RENAME TO {table.name};
                        """
                    )
                )
                # TODO: this won't deal with constraits
                # Re-apply indices
                for idx in table.indexes:
                    # TODO: can we do better? It's expensive to re-create all indices
                    idx.create(session.bind)
        except Exception as e:
            temp_table.drop(session.bind, checkfirst=True)
            raise e
