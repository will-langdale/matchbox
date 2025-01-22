from sqlalchemy.orm import Session

from matchbox.common.db import get_schema_table_names
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Resolutions,
    Sources,
)
from matchbox.server.postgresql.utils.query import (
    _build_match_query,
    _resolve_cluster_hierarchy,
    source_to_dataset_resolution,
)


def compile_query_sql(point_of_truth: str, dataset_name: str) -> str:
    """Compiles a the SQL for query() based on a single point of truth and dataset.

    Args:
        point_of_truth (string): The name of the resolution to use, like "linker_1"
        dataset_name (string): The name of the dataset to retrieve, like "dbt.companies"

    Returns:
        A compiled PostgreSQL query, including semicolon, ready to run on Matchbox
    """
    engine = MBDB.get_engine()

    source_schema, source_table = get_schema_table_names(dataset_name)

    with Session(engine) as session:
        point_of_truth_resolution = (
            session.query(Resolutions)
            .filter(Resolutions.name == point_of_truth)
            .first()
        )
        dataset_id = (
            session.query(Resolutions.resolution_id)
            .join(Sources, Sources.resolution_id == Resolutions.resolution_id)
            .filter(
                Sources.schema == source_schema,
                Sources.table == source_table,
            )
            .scalar()
        )

        id_query = _resolve_cluster_hierarchy(
            dataset_id=dataset_id,
            resolution=point_of_truth_resolution,
            threshold=None,
            engine=engine,
        )

    compiled_stmt = id_query.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )

    return str(compiled_stmt) + ";"


def compile_match_sql(source_pk: str, source_name: str, point_of_truth: str) -> str:
    """Compiles a the SQL for match() based on a single point of truth and dataset.

    Note this only tests the query that retrieves all valid matches for the supplied
    key. The actual match function goes on to merge this with the user's requested
    target table(s).

    Args:
        source_pk: The name of the primary key of the source table
        source_name: The name of the source table, like "dbt.companies"
        point_of_truth: The name of the resolution to use, like "linker_1"

    Returns:
        A compiled PostgreSQL query, including semicolon, ready to run on Matchbox
    """
    engine = MBDB.get_engine()

    with Session(engine) as session:
        source_resolution_id = source_to_dataset_resolution(
            source_name, session
        ).resolution_id

        match_query = _build_match_query(
            source_pk=source_pk,
            source_resolution_id=source_resolution_id,
            resolution=point_of_truth,
            session=session,
            threshold=None,
        )

    compiled_stmt = match_query.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )

    return str(compiled_stmt) + ";"
