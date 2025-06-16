"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import Any, Literal, TypeVar

import pyarrow as pa
from sqlalchemy import (
    ColumnElement,
    FromClause,
    and_,
    case,
    func,
    join,
    literal,
    outerjoin,
    select,
)
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ResolutionName, SourceResolutionName
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.common.sources import Match
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql

T = TypeVar("T")


def get_source_config(name: SourceResolutionName, session: Session) -> SourceConfigs:
    """Converts the named source to a SourceConfigs ORM object."""
    source_config = (
        session.query(SourceConfigs)
        .join(Resolutions, Resolutions.resolution_id == SourceConfigs.resolution_id)
        .filter(Resolutions.name == name)
        .first()
    )
    if source_config is None:
        raise MatchboxSourceNotFoundError(name=name)

    return source_config


def _resolve_thresholds(
    lineage_truths: dict[int, float | None],
    resolution: Resolutions,
    threshold: int | None,
) -> dict[int, float | None]:
    """Resolves final thresholds for each resolution in the lineage."""
    resolved_thresholds = {}

    for resolution_id, default_truth in lineage_truths.items():
        if default_truth is None:  # Source
            resolved_thresholds[resolution_id] = None
        elif threshold is None:  # Model with default threshold
            resolved_thresholds[resolution_id] = default_truth
        elif isinstance(threshold, int):  # Model with override
            resolved_thresholds[resolution_id] = (
                threshold
                if resolution_id == resolution.resolution_id
                else default_truth
            )
        else:
            raise ValueError(f"Invalid threshold type: {type(threshold)}")

    return resolved_thresholds


def _get_resolution_priority(
    resolution_id: int, context_resolution: Resolutions
) -> int:
    """Get priority level for a resolution (lower = higher priority)."""
    with MBDB.get_session() as session:
        priority_query = select(ResolutionFrom.level).where(
            and_(
                ResolutionFrom.parent == resolution_id,
                ResolutionFrom.child == context_resolution.resolution_id,
            )
        )
        level = session.execute(priority_query).scalar()
        return level if level is not None else 0


def _build_unified_query(
    resolution: Resolutions,
    resolved_thresholds: dict[int, float | None],
    source_config_filter: Any,
    columns: Literal["full", "id_key"] = "full",
) -> Select:
    """Build unified query to resolve cluster assignments across resolutions."""
    # Extract and sort model resolutions by hierarchy priority
    model_resolutions: list[tuple[int, float, int]] = [
        (res_id, threshold_val, _get_resolution_priority(res_id, resolution))
        for res_id, threshold_val in resolved_thresholds.items()
        if threshold_val is not None
    ]
    model_resolutions.sort(key=lambda x: (x[2], x[0]))  # priority, then res_id

    # Create proper aliases for clusters tables
    leaf_clusters = aliased(Clusters, name="leaf_clusters")
    root_clusters = aliased(Clusters, name="root_clusters")

    # Handle source-only case
    if not model_resolutions:
        if columns == "id_key":
            return select(
                ClusterSourceKey.cluster_id.label("id"),
                ClusterSourceKey.key,
            ).where(source_config_filter)
        else:  # "full"
            return (
                select(
                    ClusterSourceKey.cluster_id.label("root_id"),
                    leaf_clusters.cluster_hash.label("root_hash"),
                    ClusterSourceKey.cluster_id.label("leaf_id"),
                    leaf_clusters.cluster_hash.label("leaf_hash"),
                    ClusterSourceKey.key.label("leaf_key"),
                    ClusterSourceKey.source_config_id,
                )
                .select_from(
                    join(
                        ClusterSourceKey,
                        leaf_clusters,
                        ClusterSourceKey.cluster_id == leaf_clusters.cluster_id,
                    )
                )
                .where(source_config_filter)
            )

    # Build subqueries for each model resolution
    subqueries: list[Select] = []
    priority_columns: list[ColumnElement] = []
    cluster_columns: list[ColumnElement] = []

    for i, (res_id, threshold_val, _) in enumerate(model_resolutions):
        priority_value: int = 2**i  # 1, 2, 4, 8, etc.

        subquery_alias: Select = (
            select(
                Contains.leaf,
                Probabilities.cluster.label(f"cluster{i + 1}"),
                literal(priority_value).label(f"priority{i + 1}"),
            )
            .select_from(
                join(Contains, Probabilities, Contains.root == Probabilities.cluster)
            )
            .where(
                and_(
                    Probabilities.resolution == res_id,
                    Probabilities.probability >= threshold_val,
                    Probabilities.role_flag >= 1,
                )
            )
            .subquery(f"j{i + 1}")
        )

        subqueries.append(subquery_alias)
        priority_columns.append(func.coalesce(subquery_alias.c[f"priority{i + 1}"], 0))
        cluster_columns.append(subquery_alias.c[f"cluster{i + 1}"])

    # Build FROM clause - start with cluster_keys
    # Optionally join leaf clusters for hashes
    if columns == "full":
        from_clause: FromClause = join(
            ClusterSourceKey,
            leaf_clusters,
            ClusterSourceKey.cluster_id == leaf_clusters.cluster_id,
        )
    else:
        from_clause = ClusterSourceKey

    # Add LEFT JOINs for each model resolution subquery
    for subquery_alias in subqueries:
        from_clause: FromClause = outerjoin(
            from_clause,
            subquery_alias,
            ClusterSourceKey.cluster_id == subquery_alias.c.leaf,
        )

    # Build combined priority using bitwise OR
    combined_priority: ColumnElement = priority_columns[0]
    for col in priority_columns[1:]:
        combined_priority = combined_priority.op("|")(col)

    # Build CASE conditions in priority order
    # model_resolutions is sorted by priority (highest priority first)
    # Each gets a power-of-2 bit value: 1st gets bit 1, 2nd gets bit 2, etc.
    # Check bits in ascending order (1, 2, 4...) = check resolutions in priority order
    case_conditions: list[tuple[ColumnElement, ColumnElement]] = []
    for i in range(len(model_resolutions)):
        bit_value = 2**i
        case_conditions.append(
            (combined_priority.op("&")(bit_value) > 0, cluster_columns[i])
        )

    final_root_cluster_id: ColumnElement = case(
        *case_conditions, else_=ClusterSourceKey.cluster_id
    )

    # Build final query with appropriate columns
    if columns == "id_key":
        return (
            select(
                final_root_cluster_id.label("id"),
                ClusterSourceKey.key,
            )
            .select_from(from_clause)
            .where(source_config_filter)
        )
    else:  # "full"
        return (
            select(
                final_root_cluster_id.label("root_id"),
                root_clusters.cluster_hash.label("root_hash"),
                ClusterSourceKey.cluster_id.label("leaf_id"),
                leaf_clusters.cluster_hash.label("leaf_hash"),
                ClusterSourceKey.key.label("leaf_key"),
                ClusterSourceKey.source_config_id,
            )
            .select_from(
                from_clause.outerjoin(
                    root_clusters, final_root_cluster_id == root_clusters.cluster_id
                )
            )
            .where(source_config_filter)
        )


def _get_lineage_and_source_filter(
    resolution: Resolutions, sources: list[Resolutions] | None = None
) -> tuple[dict[int, float | None], Any]:
    """Get lineage truths and build source config filter."""
    if sources:
        # Get lineage to specific sources only
        lineage_truths: dict[int, float | None] = {}
        for source in sources:
            try:
                source_lineage = resolution.get_lineage_to_source(source=source)
                lineage_truths.update(source_lineage)
            except ValueError:
                continue
        if not lineage_truths:
            raise ValueError("No valid lineage found")
    else:
        # Get full lineage
        lineage_truths: dict[int, float | None] = resolution.get_lineage()

    # Build source config filter from lineage
    with MBDB.get_session() as session:
        source_config_ids: list[int] = []
        for res_id, threshold_val in lineage_truths.items():
            if threshold_val is None:  # Source resolution
                source_resolution = session.get(Resolutions, res_id)
                if source_resolution and source_resolution.source_config:
                    source_config_ids.append(
                        source_resolution.source_config.source_config_id
                    )

        source_config_filter: ColumnElement = (
            ClusterSourceKey.source_config_id.in_(source_config_ids)
            if source_config_ids
            else True
        )

    return lineage_truths, source_config_filter


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox to retrieve linked data for a source."""
    with MBDB.get_session() as session:
        source_config: SourceConfigs = get_source_config(source, session)
        source_resolution: Resolutions = session.get(
            Resolutions, source_config.resolution_id
        )

        if resolution:
            truth_resolution: Resolutions = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution)
                .first()
            )
            if truth_resolution is None:
                raise MatchboxResolutionNotFoundError(name=resolution)
        else:
            truth_resolution: Resolutions = source_resolution

        # Simple case - same resolution, no hierarchy needed
        if truth_resolution.resolution_id == source_config.resolution_id:
            id_query: Select = select(
                ClusterSourceKey.cluster_id.label("id"),
                ClusterSourceKey.key,
            ).where(ClusterSourceKey.source_config_id == source_config.source_config_id)
        else:
            # Use hierarchy resolution
            lineage_truths, _ = _get_lineage_and_source_filter(
                truth_resolution, [source_resolution]
            )
            resolved_thresholds: dict[int, float | None] = _resolve_thresholds(
                lineage_truths, truth_resolution, threshold
            )

            id_query: Select = _build_unified_query(
                resolution=truth_resolution,
                resolved_thresholds=resolved_thresholds,
                source_config_filter=ClusterSourceKey.source_config_id
                == source_config.source_config_id,
                columns="id_key",
            )

        if limit:
            id_query = id_query.limit(limit)

        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(id_query)
            logger.debug(f"Query SQL: \n {stmt}")
            return sql_to_df(stmt=stmt, connection=conn, return_type="arrow")


def get_clusters_with_leaves(
    resolution: Resolutions,
) -> dict[int, dict[str, list[dict]]]:
    """Query clusters and their leaves for all parent resolutions."""
    with MBDB.get_session() as session:
        # Get parent resolution IDs
        parent_ids: list[int] = [
            row[0]
            for row in session.execute(
                select(ResolutionFrom.parent)
                .where(ResolutionFrom.child == resolution.resolution_id)
                .where(ResolutionFrom.level == 1)
            ).all()
        ]

        if not parent_ids:
            return {}

        # For each parent, get all cluster assignments
        all_clusters: dict[int, dict] = {}
        cluster_leaves: dict[int, set[tuple[int, bytes]]] = {}

        for parent_id in parent_ids:
            parent_resolution: Resolutions = session.get(Resolutions, parent_id)
            if parent_resolution is None:
                continue

            # Get lineage and build query
            lineage_truths, source_config_filter = _get_lineage_and_source_filter(
                parent_resolution, None
            )
            resolved_thresholds: dict[int, float | None] = _resolve_thresholds(
                lineage_truths, parent_resolution, None
            )

            parent_assignments: Select = _build_unified_query(
                resolution=parent_resolution,
                resolved_thresholds=resolved_thresholds,
                source_config_filter=source_config_filter,
                columns="full",
            )

            # Execute and collect results
            for row in session.execute(parent_assignments):
                root_id = row.root_id
                if root_id not in cluster_leaves:
                    cluster_leaves[root_id] = set()
                    all_clusters[root_id] = {
                        "root_hash": row.root_hash,
                        "leaves": [],
                        "probability": None,
                    }
                cluster_leaves[root_id].add((row.leaf_id, row.leaf_hash))

        # Convert tuple sets to dict lists
        for cluster_id, leaf_tuples in cluster_leaves.items():
            all_clusters[cluster_id]["leaves"] = [
                {"leaf_id": leaf_id, "leaf_hash": leaf_hash}
                for leaf_id, leaf_hash in leaf_tuples
            ]

        return all_clusters


def match(
    key: str,
    source: SourceResolutionName,
    targets: list[SourceResolutionName],
    resolution: ResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Matches an ID in a source resolution and returns the keys in the targets."""
    with MBDB.get_session() as session:
        # Get configurations
        source_config: SourceConfigs = get_source_config(source, session)
        truth_resolution: Resolutions | None = (
            session.query(Resolutions).filter(Resolutions.name == resolution).first()
        )
        if truth_resolution is None:
            raise MatchboxResolutionNotFoundError(name=resolution)

        target_configs: list[SourceConfigs] = [
            get_source_config(target, session) for target in targets
        ]

        # Get all cluster assignments
        lineage_truths, source_config_filter = _get_lineage_and_source_filter(
            truth_resolution, None
        )
        resolved_thresholds: dict[int, float | None] = _resolve_thresholds(
            lineage_truths, truth_resolution, threshold
        )

        # Build query to find source key's cluster and all matches
        all_assignments: Select = _build_unified_query(
            resolution=truth_resolution,
            resolved_thresholds=resolved_thresholds,
            source_config_filter=source_config_filter,
            columns="full",
        ).subquery("all_assignments")

        # Find cluster for our source key
        source_cluster_query: Select = (
            select(all_assignments.c.root_id)
            .where(
                and_(
                    all_assignments.c.leaf_key == key,
                    all_assignments.c.source_config_id
                    == source_config.source_config_id,
                )
            )
            .scalar_subquery()
        )

        # Find all matches in that cluster
        matches_query: Select = select(
            all_assignments.c.root_id.label("cluster"),
            all_assignments.c.source_config_id.label("source_config"),
            all_assignments.c.leaf_key.label("key"),
        ).where(all_assignments.c.root_id == source_cluster_query)

        logger.debug(f"Match SQL: \n {compile_sql(matches_query)}")
        matches = session.execute(matches_query).all()

        # Organise matches by source config
        cluster: int | None = None
        matches_by_source_id: dict[int, set] = {}
        for cluster_id, source_config_id, key_in_source in matches:
            if cluster is None:
                cluster = cluster_id
            if source_config_id not in matches_by_source_id:
                matches_by_source_id[source_config_id] = set()
            matches_by_source_id[source_config_id].add(key_in_source)

        # Build result objects
        result: list[Match] = []
        for target, target_config in zip(targets, target_configs, strict=False):
            match_obj = Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_source_id.get(
                    source_config.source_config_id, set()
                ),
                target=target,
                target_id=matches_by_source_id.get(
                    target_config.source_config_id, set()
                ),
            )
            result.append(match_obj)

        return result
