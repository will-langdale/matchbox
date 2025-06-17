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
    union_all,
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


def _process_model_resolutions_by_priority(
    resolved_thresholds: dict[int, float | None], resolution: Resolutions
) -> list[tuple[int, float, int]]:
    """Extract and sort model resolutions by hierarchy priority."""
    model_resolutions = [
        (res_id, threshold_val, _get_resolution_priority(res_id, resolution))
        for res_id, threshold_val in resolved_thresholds.items()
        if threshold_val is not None
    ]
    return sorted(model_resolutions, key=lambda x: (x[2], x[0]))


def _build_probability_subquery(
    res_id: int, threshold_val: float, priority_value: int, subquery_name: str
) -> Select:
    """Build a probability subquery for a given resolution."""
    return (
        select(
            Contains.leaf,
            Probabilities.cluster.label(f"cluster{subquery_name}"),
            literal(priority_value).label(f"priority{subquery_name}"),
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
        .subquery(f"j{subquery_name}")
    )


def _build_bitwise_priority_case(
    subqueries: list, cluster_columns: list[ColumnElement]
) -> ColumnElement:
    """Build combined priority using bitwise OR and create CASE expression."""
    if not subqueries:
        return ClusterSourceKey.cluster_id

    # Build combined priority using bitwise OR
    priority_columns = [
        func.coalesce(subquery.c[f"priority{i + 1}"], 0)
        for i, subquery in enumerate(subqueries)
    ]
    combined_priority = priority_columns[0]
    for col in priority_columns[1:]:
        combined_priority = combined_priority.op("|")(col)

    # Build CASE conditions in priority order
    case_conditions = []
    for i in range(len(subqueries)):
        bit_value = 2**i
        case_conditions.append(
            (combined_priority.op("&")(bit_value) > 0, cluster_columns[i])
        )

    return case(*case_conditions, else_=ClusterSourceKey.cluster_id)


def _build_unified_query(
    resolution: Resolutions,
    resolved_thresholds: dict[int, float | None],
    source_config_filter: Any,
    columns: Literal["full", "id_key"] = "full",
) -> Select:
    """Build unified query to resolve cluster assignments across resolutions."""
    # Extract and sort model resolutions by hierarchy priority
    model_resolutions = _process_model_resolutions_by_priority(
        resolved_thresholds, resolution
    )

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
    cluster_columns: list[ColumnElement] = []

    for i, (res_id, threshold_val, _) in enumerate(model_resolutions):
        priority_value: int = 2**i  # 1, 2, 4, 8, etc.
        subquery = _build_probability_subquery(
            res_id, threshold_val, priority_value, str(i + 1)
        )
        subqueries.append(subquery)
        cluster_columns.append(subquery.c[f"cluster{i + 1}"])

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
    for subquery in subqueries:
        from_clause: FromClause = outerjoin(
            from_clause,
            subquery,
            ClusterSourceKey.cluster_id == subquery.c.leaf,
        )

    # Build final cluster ID using bitwise priority logic
    final_root_cluster_id = _build_bitwise_priority_case(subqueries, cluster_columns)

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


def _build_target_cluster_cte(
    key: str,
    source_config_id: int,
    resolution: Resolutions,
    threshold: int | None,
    session: Session,
) -> Select:
    """Build the target_cluster CTE using bitwise priority logic."""
    # Get lineage and resolve thresholds
    lineage_truths, _ = _get_lineage_and_source_filter(resolution, None)
    resolved_thresholds = _resolve_thresholds(lineage_truths, resolution, threshold)

    # Extract model resolutions and sort by priority
    model_resolutions = _process_model_resolutions_by_priority(
        resolved_thresholds, resolution
    )

    # Build subqueries for each resolution
    subqueries = []
    cluster_columns = []
    for i, (res_id, threshold_val, _) in enumerate(model_resolutions):
        priority_value = 2**i  # 1, 2, 4, 8, etc.
        subquery = _build_probability_subquery(
            res_id, threshold_val, priority_value, str(i + 1)
        )
        subqueries.append(subquery)
        cluster_columns.append(subquery.c[f"cluster{i + 1}"])

    # Build FROM clause starting with cluster_keys
    from_clause = ClusterSourceKey

    # Add LEFT JOINs for each resolution subquery
    for subquery in subqueries:
        from_clause = outerjoin(
            from_clause,
            subquery,
            ClusterSourceKey.cluster_id == subquery.c.leaf,
        )

    # Build final cluster ID using bitwise priority logic
    final_cluster_id = _build_bitwise_priority_case(subqueries, cluster_columns)

    return (
        select(final_cluster_id.label("cluster_id"))
        .select_from(from_clause)
        .where(
            and_(
                ClusterSourceKey.key == key,
                ClusterSourceKey.source_config_id == source_config_id,
            )
        )
    )


def _build_matching_leaves_cte(
    target_source_config_ids: list[int],
    resolution: Resolutions,
    threshold: int | None,
    target_cluster_cte,  # Pass the CTE so we can reference it properly
    session: Session,
) -> Select:
    """Build the matching_leaves CTE with UNION ALL branches."""
    # Get lineage and resolve thresholds
    lineage_truths, _ = _get_lineage_and_source_filter(resolution, None)
    resolved_thresholds = _resolve_thresholds(lineage_truths, resolution, threshold)

    # Extract model resolutions
    model_resolutions = [
        (res_id, threshold_val)
        for res_id, threshold_val in resolved_thresholds.items()
        if threshold_val is not None
    ]

    # Start with direct members branch
    branches = [
        select(
            ClusterSourceKey.cluster_id,
            ClusterSourceKey.key,
            ClusterSourceKey.source_config_id,
        )
        .select_from(
            ClusterSourceKey, target_cluster_cte
        )  # CROSS JOIN with target_cluster
        .where(
            and_(
                ClusterSourceKey.source_config_id.in_(target_source_config_ids),
                ClusterSourceKey.cluster_id == target_cluster_cte.c.cluster_id,
            )
        )
    ]

    # Add branches for each model resolution
    for res_id, threshold_val in model_resolutions:
        branch = (
            select(
                ClusterSourceKey.cluster_id,
                ClusterSourceKey.key,
                ClusterSourceKey.source_config_id,
            )
            .select_from(
                join(
                    join(
                        ClusterSourceKey,
                        Contains,
                        ClusterSourceKey.cluster_id == Contains.leaf,
                    ),
                    Probabilities,
                    and_(
                        Contains.root == Probabilities.cluster,
                        Probabilities.resolution == res_id,
                        Probabilities.probability >= threshold_val,
                        Probabilities.role_flag >= 1,
                    ),
                ),
                target_cluster_cte,  # CROSS JOIN with target_cluster
            )
            .where(
                and_(
                    ClusterSourceKey.source_config_id.in_(target_source_config_ids),
                    Probabilities.cluster == target_cluster_cte.c.cluster_id,
                )
            )
        )
        branches.append(branch)

    # Combine all branches with UNION ALL
    if len(branches) == 1:
        return branches[0]

    # Use SQLAlchemy 2.0 union_all function
    return union_all(*branches)


def _build_match_query(
    key: str,
    source_config_id: int,
    target_source_config_ids: list[int],
    resolution: Resolutions,
    threshold: int | None,
    session: Session,
) -> Select:
    """Combine CTEs into the final match query."""
    target_cluster_cte = _build_target_cluster_cte(
        key, source_config_id, resolution, threshold, session
    ).cte("target_cluster")

    # Include both source and target configs to get all keys in the cluster
    all_source_config_ids = [source_config_id] + target_source_config_ids
    matching_leaves_cte = _build_matching_leaves_cte(
        all_source_config_ids, resolution, threshold, target_cluster_cte, session
    ).cte("matching_leaves")

    # LEFT JOIN to ensure we always get the cluster even if no target matches
    return (
        select(
            target_cluster_cte.c.cluster_id.label("cluster"),
            matching_leaves_cte.c.source_config_id,
            matching_leaves_cte.c.key,
        )
        .select_from(
            target_cluster_cte.outerjoin(
                matching_leaves_cte,
                literal(True),  # Always join - we want all target_cluster rows
            )
        )
        .distinct()
    )


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox to retrieve linked data for a source.

    Retrieves all linked data for a given source, resolving through hierarchy if needed.

    * Simple case: If querying the same resolution as the source, just select cluster
        IDs and keys directly from ClusterSourceKey
    * Hierarchy case: Uses the unified query builder to traverse up the resolution
        hierarchy, applying bitwise priority logic to determine which parent cluster
        each source record belongs to
    * Priority resolution: When multiple model resolutions could assign a record to
        different clusters, uses powers-of-2 bit flags to ensure higher-priority
        resolutions win

    Returns all records with their final resolved cluster IDs.
    """
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
    """Query clusters and their leaves for all parent resolutions.

    For a given resolution, find all its parent resolutions and return complete
    cluster compositions.

    * Parent discovery: Queries ResolutionFrom to find all direct parent
        resolutions (level 1)
    * Cluster building: For each parent, runs the full unified query to get all
        cluster assignments with both root and leaf information
    * Aggregation: Collects all leaf nodes belonging to each root cluster across all
        parent resolutions

    Return a dictionary mapping cluster IDs to their complete leaf compositions
    and metadata.
    """
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
    """Matches an ID in a source resolution and returns the keys in the targets.

    Given a specific key in a source, find what it matches to in target sources
    through a resolution hierarchy.

    * Target cluster identification: Uses bitwise priority CTE to determine which
        cluster the input key belongs to at the resolution level
    * Matching leaves discovery: Builds UNION ALL query with branches for:
        * Direct cluster members (source-only case)
        * Members connected through each model resolution in the hierarchy
    * Cross-reference: Joins the target cluster with all possible matching leaves,
        filtering for the requested target sources

    Organises matches by source configuration and returns structured Match objects
    for each target.
    """
    with MBDB.get_session() as session:
        # Get configurations
        source_config: SourceConfigs = get_source_config(source, session)
        truth_resolution: Resolutions | None = session.execute(
            select(Resolutions).where(Resolutions.name == resolution)
        ).scalar()
        if truth_resolution is None:
            raise MatchboxResolutionNotFoundError(name=resolution)

        target_configs: list[SourceConfigs] = [
            get_source_config(target, session) for target in targets
        ]
        target_source_config_ids = [tc.source_config_id for tc in target_configs]

        # Build and execute the match query
        matches_query = _build_match_query(
            key,
            source_config.source_config_id,
            target_source_config_ids,
            truth_resolution,
            threshold,
            session,
        )

        logger.debug(f"Match SQL: \n {compile_sql(matches_query)}")
        matches = session.execute(matches_query).all()

        # Organise matches by source config
        cluster: int | None = None
        matches_by_source_id: dict[int, set] = {}

        for cluster_id, source_config_id_result, key_in_source in matches:
            if cluster is None:
                cluster = cluster_id

            # Skip NULL results from LEFT JOIN (when no target matches)
            if source_config_id_result is not None and key_in_source is not None:
                if source_config_id_result not in matches_by_source_id:
                    matches_by_source_id[source_config_id_result] = set()
                matches_by_source_id[source_config_id_result].add(key_in_source)

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
