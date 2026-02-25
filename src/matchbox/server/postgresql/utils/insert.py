"""Utilities for inserting data into the PostgreSQL backend."""

import pyarrow as pa
from sqlalchemy import exists, func, join, literal, select
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BYTEA,
    SMALLINT,
    TEXT,
    aggregate_order_by,
    insert,
)
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import TableClause

from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.db import QueryReturnType, sql_to_df
from matchbox.common.dtos import (
    ModelResolutionPath,
    ResolutionType,
    ResolverResolutionPath,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxResolutionExistingData,
    MatchboxResolutionInvalidData,
)
from matchbox.common.hash import hash_arrow_table, hash_model_results
from matchbox.common.logging import logger
from matchbox.common.transform import hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    ModelEdges,
    ResolutionClusters,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql, ingest_to_temporary_table


def insert_hashes(
    path: SourceResolutionPath, data_hashes: pa.Table, batch_size: int
) -> None:
    """Indexes hash data for a source.

    Args:
        path: The path of the source resolution
        data_hashes: Arrow table containing hash data
        batch_size: Batch size for bulk operations

    Raises:
        MatchboxResolutionNotFoundError: If the specified resolution doesn't exist.
        MatchboxResolutionInvalidData: If data fingerprint conflicts with resolution.
        MatchboxResolutionExistingData: If data was already inserted for resolution.
    """
    log_prefix = f"Index hashes {path}"
    if data_hashes.num_rows == 0:
        logger.info("No hashes given.", prefix=log_prefix)
        return

    fingerprint = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        resolution = Resolutions.from_path(
            path=path, res_type=ResolutionType.SOURCE, session=session
        )
        # Check if the content hash is the same
        if resolution.fingerprint != fingerprint:
            raise MatchboxResolutionInvalidData

        # Determine if the resolution already has any keys
        existing_keys = session.execute(
            select(func.count())
            .select_from(
                join(
                    ClusterSourceKey,
                    SourceConfigs,
                    ClusterSourceKey.source_config_id == SourceConfigs.source_config_id,
                )
            )
            .where(SourceConfigs.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing_keys > 0:
            raise MatchboxResolutionExistingData

        source_config_id = resolution.source_config.source_config_id

    with (
        ingest_to_temporary_table(
            table_name="incoming_hashes",
            schema_name="mb",
            column_types={"hash": BYTEA(), "keys": ARRAY(TEXT)},
            data=data_hashes.select(["hash", "keys"]),
            max_chunksize=batch_size,
        ) as incoming,
        MBDB.get_session() as session,
    ):
        try:
            # Add clusters
            new_hashes = (
                select(incoming.c.hash)
                .distinct()
                .where(
                    ~exists(select(1).where(Clusters.cluster_hash == incoming.c.hash))
                )
            )

            stmt_insert_clusters = (
                insert(Clusters)
                .from_select(["cluster_hash"], new_hashes)
                .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
            )

            result = session.execute(stmt_insert_clusters)
            logger.info(
                f"Will add {result.rowcount:,} entries to Clusters table",
                prefix=log_prefix,
            )
            session.flush()

            # Add source keys
            exploded = select(
                Clusters.cluster_id,
                literal(source_config_id, BIGINT).label("source_config_id"),
                func.unnest(incoming.c["keys"]).label("key"),
            ).select_from(
                incoming.join(Clusters, Clusters.cluster_hash == incoming.c.hash)
            )

            stmt_insert_keys = insert(ClusterSourceKey).from_select(
                ["cluster_id", "source_config_id", "key"],
                exploded,
            )

            result = session.execute(stmt_insert_keys)
            logger.info(
                f"Will add {result.rowcount:,} entries to ClusterSourceKey table",
                prefix=log_prefix,
            )
            session.commit()

        except Exception as e:
            # Log the error and rollback
            logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
            session.rollback()
            raise

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        ClusterSourceKey.__table__.fullname,
    )

    logger.info("Finished", prefix=log_prefix)


def insert_model_edges(
    path: ModelResolutionPath,
    results: pa.Table,
    batch_size: int,
) -> None:
    """Writes model edges to Matchbox."""
    log_prefix = f"Model {path.name}"
    if results.num_rows == 0:
        logger.info("Empty model edges given.", prefix=log_prefix)
        return

    resolution = Resolutions.from_path(path=path, res_type=ResolutionType.MODEL)

    # Check if the content hash is the same
    fingerprint = hash_model_results(results)
    if resolution.fingerprint != fingerprint:
        raise MatchboxResolutionInvalidData

    with MBDB.get_session() as session:
        existing_edges = session.execute(
            select(func.count())
            .select_from(ModelEdges)
            .where(ModelEdges.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing_edges > 0:
            raise MatchboxResolutionExistingData

    logger.info(
        f"Writing model edges with batch size {batch_size:,}", prefix=log_prefix
    )

    with (
        MBDB.get_session() as session,
        ingest_to_temporary_table(
            table_name="incoming_model_edges",
            schema_name="mb",
            column_types={
                "left_id": BIGINT(),
                "right_id": BIGINT(),
                "probability": SMALLINT(),
            },
            data=results,
            max_chunksize=batch_size,
        ) as incoming_edges,
    ):
        try:
            edges_select = select(
                literal(resolution.resolution_id, BIGINT).label("resolution_id"),
                incoming_edges.c.left_id,
                incoming_edges.c.right_id,
                incoming_edges.c.probability,
            )
            inserted = session.execute(
                insert(ModelEdges).from_select(
                    ["resolution_id", "left_id", "right_id", "probability"],
                    edges_select,
                )
            )
            logger.info(
                f"Will add {inserted.rowcount:,} entries to ModelEdges table",
                prefix=log_prefix,
            )
            session.commit()

        except Exception as e:
            logger.error(
                f"Failed to insert model edges, rolling back: {str(e)}",
                prefix=log_prefix,
            )
            session.rollback()
            raise

    MBDB.vacuum_analyze(ModelEdges.__table__.fullname)
    logger.info("Model edge insert complete!", prefix=log_prefix)


def _compute_resolver_hashes(
    incoming_cluster_assignments: TableClause,
    session: Session,
) -> pa.Table:
    """Expand cluster assignments to leaves and compute deterministic cluster hashes.

    Each uploaded assignment maps a ``client_cluster_id`` to a
    ``server_cluster_id``. A server cluster may itself be a parent cluster (with
    children in ``Contains``), or a leaf cluster with no children. This function:

    1. Expands each server cluster to its leaf-level cluster IDs via an outer
        join on ``Contains``. Clusters with no children resolve to themselves.
    2. Groups the leaf hashes per client cluster, sorted for determinism.
    3. Computes a single composite hash per client cluster in Python.

    Uses an **inner join** to ``Clusters`` when fetching hashes, so unknown
    leaf IDs are silently dropped here. They will surface later as FK
    violations when inserting into ``Contains``.

    Args:
        incoming_cluster_assignments: Temporary table with
            (client_cluster_id, server_cluster_id).
        session: Active SQLAlchemy session (must share the temp table's connection).

    Returns:
        Arrow table with columns (client_cluster_id: int64, cluster_hash: binary).
    """
    # Expand each server_cluster_id to its constituent leaves
    expanded_leaves = (
        select(
            incoming_cluster_assignments.c.client_cluster_id,
            func.coalesce(
                Contains.leaf,
                incoming_cluster_assignments.c.server_cluster_id,
            ).label("leaf_id"),
        )
        .distinct()
        .select_from(
            incoming_cluster_assignments.outerjoin(
                Contains,
                Contains.root == incoming_cluster_assignments.c.server_cluster_id,
            )
        )
        .subquery("expanded_leaves")
    )

    # Group leaf hashes per client cluster, sorted for deterministic hashing
    rows = session.execute(
        select(
            expanded_leaves.c.client_cluster_id,
            func.array_agg(
                aggregate_order_by(
                    Clusters.cluster_hash,
                    Clusters.cluster_hash,
                )
            ).label("leaf_hashes"),
        )
        .select_from(
            expanded_leaves.join(
                Clusters,
                Clusters.cluster_id == expanded_leaves.c.leaf_id,
            )
        )
        .group_by(expanded_leaves.c.client_cluster_id)
        .order_by(expanded_leaves.c.client_cluster_id)
    ).all()

    # Compute a single composite hash per client cluster from its sorted leaves
    return pa.table(
        {
            "client_cluster_id": [int(r[0]) for r in rows],
            "cluster_hash": [
                hash_cluster_leaves([bytes(h) for h in r[1]]) for r in rows
            ],
        }
    )


def insert_resolver_clusters(
    path: ResolverResolutionPath,
    cluster_assignments: pa.Table,
    batch_size: int,
) -> pa.Table:
    """Write resolver cluster assignments and return a client-to-server mapping.

    The function proceeds in three phases:

    1. Validate: check fingerprint, ensure no prior data, short-circuit
        if the upload is empty
    2. Compute hashes: ingest assignments to a temp table, expand each
        server cluster to its leaves, and derive a deterministic cluster hash per client
        cluster (Python round-trip via ``_compute_resolver_hashes``)
    3. Insert everything: with both temp tables live in one session,
        materialise new ``Clusters`` rows, then insert ``Contains`` and
        ``ResolutionClusters`` membership rows, and return the mapping

    Args:
        path: The resolver resolution path to upload cluster assignments for
        cluster_assignments: Arrow table with
            (client_cluster_id, server_cluster_id) columns
        batch_size: Batch size for temporary table ingestion

    Returns:
        Arrow table with (client_cluster_id, server_cluster_id), where
        ``server_cluster_id`` is the canonical Matchbox cluster ID,
        conforming to ``SCHEMA_CLUSTERS``

    Raises:
        MatchboxResolutionNotFoundError: If the resolution doesn't exist
        MatchboxResolutionInvalidData: If the fingerprint doesn't match
        MatchboxResolutionExistingData: If clusters already exist for this resolver
    """
    log_prefix = f"Resolver {path.name}"
    fingerprint = hash_arrow_table(cluster_assignments)
    cluster_assignment_data = cluster_assignments.select(
        ["client_cluster_id", "server_cluster_id"]
    )
    mapping: pa.Table

    # 1) Validate

    with MBDB.get_session() as session:
        resolution = Resolutions.from_path(
            path=path, res_type=ResolutionType.RESOLVER, session=session
        )

        if resolution.fingerprint != fingerprint:
            raise MatchboxResolutionInvalidData

        existing = session.execute(
            select(func.count())
            .select_from(ResolutionClusters)
            .where(ResolutionClusters.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing > 0:
            raise MatchboxResolutionExistingData

        if cluster_assignment_data.num_rows == 0:
            return pa.Table.from_pydict(
                {"client_cluster_id": [], "server_cluster_id": []},
                schema=SCHEMA_CLUSTERS,
            )

        resolution_id = resolution.resolution_id

    # 2) Compute hashes
    #
    # The assignments temp table stays open for both phases: we query it to
    # compute hashes, then reference it again for the Contains insert. The
    # hashes temp table is opened inside once the Arrow data is ready
    with (
        ingest_to_temporary_table(
            table_name="resolver_assignments",
            schema_name="mb",
            column_types={
                "client_cluster_id": BIGINT(),
                "server_cluster_id": BIGINT(),
            },
            data=cluster_assignment_data,
            max_chunksize=batch_size,
        ) as incoming_cluster_assignments,
        MBDB.get_session() as session,
    ):
        # Expand server clusters → leaves → deterministic hash per client cluster
        hash_data = _compute_resolver_hashes(incoming_cluster_assignments, session)

        # Leaf expansion subquery
        expanded_leaves = (
            select(
                incoming_cluster_assignments.c.client_cluster_id,
                func.coalesce(
                    Contains.leaf,
                    incoming_cluster_assignments.c.server_cluster_id,
                ).label("leaf_id"),
            )
            .distinct()
            .select_from(
                # Clusters with no children in Contains resolve to themselves
                incoming_cluster_assignments.outerjoin(
                    Contains,
                    Contains.root == incoming_cluster_assignments.c.server_cluster_id,
                )
            )
            .subquery("expanded_leaves")
        )

        # 3) Insert everything
        with ingest_to_temporary_table(
            table_name="resolver_hashes",
            schema_name="mb",
            column_types={"client_cluster_id": BIGINT(), "cluster_hash": BYTEA()},
            data=hash_data,
            max_chunksize=batch_size,
        ) as incoming_hashes:
            try:
                # Clusters
                # Insert any hashes we haven't seen before
                session.execute(
                    insert(Clusters)
                    .from_select(
                        ["cluster_hash"],
                        select(incoming_hashes.c.cluster_hash)
                        .distinct()
                        .where(
                            ~exists(
                                select(1).where(
                                    Clusters.cluster_hash
                                    == incoming_hashes.c.cluster_hash
                                )
                            )
                        ),
                    )
                    .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
                )
                session.flush()

                # Map each client cluster to its canonical Clusters.cluster_id
                # by joining hashes back to the now-populated Clusters table
                cluster_map = (
                    select(
                        incoming_hashes.c.client_cluster_id,
                        Clusters.cluster_id,
                    )
                    .select_from(
                        incoming_hashes.join(
                            Clusters,
                            Clusters.cluster_hash == incoming_hashes.c.cluster_hash,
                        )
                    )
                    .subquery("cluster_map")
                )

                # Contains
                # Record which leaves belong to each new resolver cluster
                session.execute(
                    insert(Contains)
                    .from_select(
                        ["root", "leaf"],
                        select(
                            cluster_map.c.cluster_id,
                            expanded_leaves.c.leaf_id,
                        )
                        .select_from(
                            expanded_leaves.join(
                                cluster_map,
                                expanded_leaves.c.client_cluster_id
                                == cluster_map.c.client_cluster_id,
                            )
                        )
                        .distinct(),
                    )
                    .on_conflict_do_nothing(
                        index_elements=[Contains.root, Contains.leaf]
                    )
                )

                # ResolutionClusters
                # Associate all proposed clusters with this resolver resolution
                session.execute(
                    insert(ResolutionClusters)
                    .from_select(
                        ["resolution_id", "cluster_id"],
                        select(
                            literal(resolution_id, BIGINT).label("resolution_id"),
                            cluster_map.c.cluster_id,
                        ).distinct(),
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            ResolutionClusters.resolution_id,
                            ResolutionClusters.cluster_id,
                        ]
                    )
                )

                session.commit()

                # Mapping
                # Read back the client→canonical mapping
                mapping_query = select(
                    cluster_map.c.client_cluster_id,
                    cluster_map.c.cluster_id.label("server_cluster_id"),
                ).order_by(cluster_map.c.client_cluster_id)

                with MBDB.get_adbc_connection() as connection:
                    mapping = sql_to_df(
                        stmt=compile_sql(mapping_query),
                        connection=connection,
                        return_type=QueryReturnType.ARROW,
                    )

            except Exception:
                session.rollback()
                raise

    logger.info("Resolver cluster insert complete!", prefix=log_prefix)

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        Contains.__table__.fullname,
        ResolutionClusters.__table__.fullname,
    )

    return mapping
