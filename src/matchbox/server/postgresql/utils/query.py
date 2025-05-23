"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import TypeVar

import pyarrow as pa
from sqlalchemy import BIGINT, and_, cast, func, literal, null, select, union
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import CTE, Select

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
    lineage_truths: dict[str, float],
    resolution: Resolutions,
    threshold: int | None,
) -> dict[int, float]:
    """Resolves final thresholds for each resolution in the lineage based on user input.

    Args:
        lineage_truths: Dict from with resolution hash -> cached truth
        resolution: The target resolution being used for clustering
        threshold: User-supplied threshold value

    Returns:
        Dict mapping resolution hash to their final threshold values
    """
    resolved_thresholds = {}

    for resolution_id, default_truth in lineage_truths.items():
        # Source
        if default_truth is None:
            resolved_thresholds[resolution_id] = None
            continue

        # Model
        if threshold is None:
            resolved_thresholds[resolution_id] = default_truth
        elif isinstance(threshold, int):
            resolved_thresholds[resolution_id] = (
                threshold
                if resolution_id == resolution.resolution_id
                else default_truth
            )
        else:
            raise ValueError(f"Invalid threshold type: {type(threshold)}")

    return resolved_thresholds


def _get_valid_clusters_for_resolution(resolution_id: int, threshold: int) -> Select:
    """Get clusters that meet the threshold for a specific resolution."""
    return select(Probabilities.cluster.label("cluster")).where(
        and_(
            Probabilities.resolution == resolution_id,
            Probabilities.probability >= threshold,
        )
    )


def _union_valid_clusters(lineage_thresholds: dict[int, float]) -> Select:
    """Creates a CTE of clusters that are valid for any resolution in the lineage.

    Each resolution may have a different threshold.
    """
    valid_clusters = None

    for resolution_id, threshold in lineage_thresholds.items():
        if threshold is None:
            # This is a source resolution
            # Get all its clusters through ClusterSourceKey
            resolution_valid = (
                select(ClusterSourceKey.cluster_id.label("cluster"))
                .join(
                    SourceConfigs,
                    SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
                )
                .where(SourceConfigs.resolution_id == resolution_id)
                .distinct()
            )
        else:
            # This is a model - get clusters meeting threshold
            resolution_valid = _get_valid_clusters_for_resolution(
                resolution_id, threshold
            )

        if valid_clusters is None:
            valid_clusters = resolution_valid
        else:
            valid_clusters = union(valid_clusters, resolution_valid)

    if valid_clusters is None:
        # Handle empty lineage case
        return select(cast(null(), BIGINT).label("cluster")).where(False)

    return valid_clusters.cte("valid_clusters").prefix_with("MATERIALIZED")


def _build_valid_contains(valid_clusters_cte: CTE, name: str) -> CTE:
    """Filters the Contains table to only include relationships with valid parents."""
    valid_contains = select(Contains.child, Contains.parent).where(
        Contains.parent.in_(select(valid_clusters_cte.c.cluster))
    )

    return valid_contains.cte(name).prefix_with("MATERIALIZED")


def _resolve_hierarchy_assignments(
    resolution: Resolutions,
    sources: list[Resolutions] | None = None,
    threshold: int | None = None,
) -> Select:
    """Resolve cluster assignments across a resolution's lineage.

    Args:
        resolution: Single resolution to start from
        sources: Optional list of sources to filter to. If None, includes all sources
            in lineage
        threshold: Optional threshold override

    Returns:
        Select query with columns:
        - root_id: The root cluster ID assigned by highest priority resolution
        - root_hash: Hash of the assigned root cluster
        - leaf_id: The leaf cluster ID
        - leaf_hash: Hash of the leaf cluster
        - leaf_key: The original source key
        - source_config_id: Which source config this key belongs to
    """
    # Get lineage - if sources specified, lineage to each source; otherwise full lineage
    if sources:
        # Get lineage to specific sources
        lineage_truths = {}
        for source in sources:
            try:
                source_lineage = resolution.get_lineage_to_source(source=source)
                lineage_truths.update(source_lineage)
            except ValueError:
                continue  # Skip sources not reachable from this resolution
    else:
        # Get full lineage to discover all available sources
        lineage_truths = resolution.get_lineage()

    if not lineage_truths:
        # Return empty result if no valid lineage
        return select(
            literal(None).label("root_id"),
            literal(None).label("root_hash"),
            literal(None).label("leaf_id"),
            literal(None).label("leaf_hash"),
            literal(None).label("leaf_key"),
            literal(None).label("source_config_id"),
        ).where(False)

    # Resolve thresholds for each resolution in lineage
    thresholds = _resolve_thresholds(
        lineage_truths=lineage_truths,
        resolution=resolution,
        threshold=threshold,
    )

    # Separate source and model resolutions from lineage
    source_resolutions = []
    model_resolutions = []

    for resolution_id, threshold_val in thresholds.items():
        if threshold_val is None:
            source_resolutions.append(resolution_id)
        else:
            model_resolutions.append((resolution_id, threshold_val))

    # Filter to specific source configs if sources provided
    source_config_filter = True
    if sources:
        source_config_ids = []
        with MBDB.get_session() as session:
            for source in sources:
                source_configs = (
                    session.execute(
                        select(SourceConfigs.source_config_id).where(
                            SourceConfigs.resolution_id == source.resolution_id
                        )
                    )
                    .scalars()
                    .all()
                )
                source_config_ids.extend(source_configs)

        if source_config_ids:
            source_config_filter = ClusterSourceKey.source_config_id.in_(
                source_config_ids
            )

    decisions = []

    # Get source decisions (type = 'source')
    if source_resolutions:
        source_decisions = (
            select(
                ClusterSourceKey.key.label("leaf_key"),
                Clusters.cluster_id.label("leaf_id"),
                Clusters.cluster_hash.label("leaf_hash"),
                Clusters.cluster_id.label("root_id"),  # No hierarchy for sources
                Clusters.cluster_hash.label("root_hash"),
                ClusterSourceKey.source_config_id.label("source_config_id"),
                SourceConfigs.resolution_id.label("deciding_resolution_id"),
                literal(999).label("priority"),  # Source has lowest priority
            )
            .select_from(ClusterSourceKey)
            .join(Clusters, ClusterSourceKey.cluster_id == Clusters.cluster_id)
            .join(
                SourceConfigs,
                SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
            )
            .join(Resolutions, Resolutions.resolution_id == SourceConfigs.resolution_id)
            .where(
                and_(
                    Resolutions.type == "source",
                    SourceConfigs.resolution_id.in_(source_resolutions),
                    source_config_filter,
                )
            )
        )
        decisions.append(source_decisions)

    # Get model decisions (type = 'model')
    for model_resolution_id, threshold_val in model_resolutions:
        # Build threshold conditions
        prob_conditions = [
            Probabilities.resolution == model_resolution_id,
            Probabilities.role_flag >= 1,
        ]
        if threshold_val is not None:
            prob_conditions.append(Probabilities.probability >= threshold_val)

        # Create aliases for this model resolution query
        RootClusters = Clusters.__table__.alias("root_clusters")
        ModelResolutions = Resolutions.__table__.alias("model_res")

        model_decisions = (
            select(
                ClusterSourceKey.key.label("leaf_key"),
                Clusters.cluster_id.label("leaf_id"),
                Clusters.cluster_hash.label("leaf_hash"),
                func.coalesce(Contains.root, Clusters.cluster_id).label("root_id"),
                func.coalesce(RootClusters.c.cluster_hash, Clusters.cluster_hash).label(
                    "root_hash"
                ),
                ClusterSourceKey.source_config_id.label("source_config_id"),
                literal(model_resolution_id).label("deciding_resolution_id"),
                ResolutionFrom.level.label("priority"),
            )
            .select_from(ClusterSourceKey)
            .join(Clusters, ClusterSourceKey.cluster_id == Clusters.cluster_id)
            .join(
                SourceConfigs,
                SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
            )
            # Connect to hierarchy
            .outerjoin(Contains, Contains.leaf == Clusters.cluster_id)
            .outerjoin(RootClusters, Contains.root == RootClusters.c.cluster_id)
            # Connect to model decisions
            .join(
                Probabilities,
                Probabilities.cluster
                == func.coalesce(Contains.root, Clusters.cluster_id),
            )
            .join(
                ModelResolutions,
                ModelResolutions.c.resolution_id == Probabilities.resolution,
            )
            .join(
                ResolutionFrom,
                and_(
                    ResolutionFrom.parent == model_resolution_id,
                    ResolutionFrom.child == resolution.resolution_id,
                ),
            )
            .where(
                and_(
                    ModelResolutions.c.type == "model",
                    source_config_filter,
                    *prob_conditions,
                )
            )
        )
        decisions.append(model_decisions)

    # Combine all decisions
    if not decisions:
        return select(
            literal(None).label("root_id"),
            literal(None).label("root_hash"),
            literal(None).label("leaf_id"),
            literal(None).label("leaf_hash"),
            literal(None).label("leaf_key"),
            literal(None).label("source_config_id"),
        ).where(False)

    all_decisions = decisions[0]
    for decision in decisions[1:]:
        all_decisions = union(all_decisions, decision)

    all_decisions = all_decisions.cte("all_decisions")

    # For each source key, pick the decision with highest priority
    # (lowest priority number)
    ranked_decisions = (
        select(
            all_decisions.c.leaf_key,
            all_decisions.c.leaf_id,
            all_decisions.c.leaf_hash,
            all_decisions.c.root_id,
            all_decisions.c.root_hash,
            all_decisions.c.source_config_id,
            func.row_number()
            .over(
                partition_by=all_decisions.c.leaf_key,
                order_by=[
                    all_decisions.c.priority.asc(),
                    all_decisions.c.deciding_resolution_id.desc(),
                ],
            )
            .label("rank"),
        )
    ).cte("ranked_decisions")

    # Return only the highest priority decision for each key
    return select(
        ranked_decisions.c.root_id,
        ranked_decisions.c.root_hash,
        ranked_decisions.c.leaf_id,
        ranked_decisions.c.leaf_hash,
        ranked_decisions.c.leaf_key,
        ranked_decisions.c.source_config_id,
    ).where(ranked_decisions.c.rank == 1)


def _get_cluster_hierarchy_base_query() -> Select:
    """Get the base query for cluster hierarchy relationships.

    Returns a Select query with columns:
    - root_id: The root cluster ID (from Contains.root, or cluster itself)
    - root_hash: Hash of the root cluster
    - leaf_id: The leaf cluster ID (original cluster)
    - leaf_hash: Hash of the leaf cluster
    - probability: Probability from Probabilities (None for source clusters)
    - source_config_id: Which source this leaf cluster belongs to
    - resolution_id: Which resolution this relates to
    """
    RootClusters = Clusters.__table__.alias("root_clusters")

    return (
        select(
            func.coalesce(Contains.root, Clusters.cluster_id).label("root_id"),
            func.coalesce(RootClusters.c.cluster_hash, Clusters.cluster_hash).label(
                "root_hash"
            ),
            Clusters.cluster_id.label("leaf_id"),
            Clusters.cluster_hash.label("leaf_hash"),
            Probabilities.probability.label("probability"),
            ClusterSourceKey.source_config_id.label("source_config_id"),
            func.coalesce(Probabilities.resolution, SourceConfigs.resolution_id).label(
                "resolution_id"
            ),
        )
        .select_from(Clusters)
        # Connect to source keys
        .join(ClusterSourceKey, ClusterSourceKey.cluster_id == Clusters.cluster_id)
        .join(
            SourceConfigs,
            SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
        )
        # Connect to hierarchy (may be empty)
        .outerjoin(Contains, Contains.leaf == Clusters.cluster_id)
        .outerjoin(RootClusters, Contains.root == RootClusters.c.cluster_id)
        # Connect to model decisions (may be empty for source resolutions)
        .outerjoin(
            Probabilities,
            Probabilities.cluster == func.coalesce(Contains.root, Clusters.cluster_id),
        )
    )


def _resolve_cluster_hierarchy(
    source_config: SourceConfigs,
    truth_resolution: Resolutions,
    threshold: int | None = None,
) -> Select:
    """Resolves the final cluster assignments for all records in a source.

    Args:
        source_config: SourceConfig object of the source to query
        truth_resolution: Resolution object representing the point of truth
        threshold: Optional threshold value

    Returns:
        SQLAlchemy Select statement that will resolve to (id, key) pairs, where
        id is the highest-priority cluster id and key is the original record key
    """
    with MBDB.get_session() as session:
        source_resolution = session.get(Resolutions, source_config.resolution_id)
        if source_resolution is None:
            raise MatchboxSourceNotFoundError()

        # If truth_resolution is the same as source resolution,
        # just return clusters directly
        if truth_resolution.resolution_id == source_config.resolution_id:
            return (
                select(
                    ClusterSourceKey.cluster_id.label("id"),
                    ClusterSourceKey.key.label("key"),
                )
                .select_from(ClusterSourceKey)
                .where(
                    ClusterSourceKey.source_config_id == source_config.source_config_id
                )
            )

        # Use the common abstraction to resolve hierarchy assignments
        hierarchy_assignments = _resolve_hierarchy_assignments(
            resolution=truth_resolution,
            sources=[source_resolution],
            threshold=threshold,
        ).cte("hierarchy_assignments")

        # Filter to our specific source config and return in the expected format
        return select(
            hierarchy_assignments.c.root_id.label("id"),
            hierarchy_assignments.c.leaf_key.label("key"),
        ).where(
            hierarchy_assignments.c.source_config_id == source_config.source_config_id
        )


def _get_clusters_with_leaves(
    resolution: Resolutions,
) -> dict[int, dict[str, list[dict]]]:
    """Query clusters and their leaves for all parent resolutions.

    Args:
        resolution: Resolution object whose parents proposed the clusters
            we need to recover

    Returns:
        Dict mapping cluster_id to a dict with cluster info and leaves list
    """
    with MBDB.get_session() as session:
        # Get parent resolution IDs
        parent_ids = [
            row[0]
            for row in session.execute(
                select(ResolutionFrom.parent)
                .where(ResolutionFrom.child == resolution.resolution_id)
                .where(ResolutionFrom.level == 1)
            ).all()
        ]

        if not parent_ids:
            return {}

        # For each parent, get all cluster assignments it endorses
        all_clusters = {}

        for parent_id in parent_ids:
            parent_resolution = session.get(Resolutions, parent_id)
            if parent_resolution is None:
                continue

            # Use the common abstraction to get all assignments this parent endorses
            parent_assignments = _resolve_hierarchy_assignments(
                resolution=parent_resolution,
                sources=None,  # All sources in parent's lineage
                threshold=None,  # Use default thresholds
            )

            # Execute and collect results for this parent
            for row in session.execute(parent_assignments):
                root_id = row.root_id

                if root_id not in all_clusters:
                    all_clusters[root_id] = {
                        "root_hash": row.root_hash,
                        "leaves": [],
                        "probability": None,  # Could get from base query if needed
                    }

                all_clusters[root_id]["leaves"].append(
                    {"leaf_id": row.leaf_id, "leaf_hash": row.leaf_hash}
                )

        return all_clusters


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox and the SourceConfig warehouse to retrieve linked data.

    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a "point of truth" resolution is supplied, will
    attach the clusters this data belongs to.

    To accomplish this, the function:

    * Iterates through each selector, and
        * Retrieves its data in Matchbox according to the optional point of truth,
        including its hash and cluster hash
        * Retrieves its raw data from its SourceConfig's warehouse
        * Joins the two together
    * Unions the results, one row per item of data in the warehouses

    Returns:
        A table containing the requested data from each table, unioned together,
        with the hash key of each row in Matchbox
    """
    with MBDB.get_session() as session:
        source_config = get_source_config(source, session)
        source_resolution = session.get(Resolutions, source_config.resolution_id)

        if resolution:
            truth_resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution)
                .first()
            )
            if truth_resolution is None:
                raise MatchboxResolutionNotFoundError(name=resolution)
        else:
            truth_resolution = source_resolution

        id_query = _resolve_cluster_hierarchy(
            source_config=source_config,
            truth_resolution=truth_resolution,
            threshold=threshold,
        )

        if limit:
            id_query = id_query.limit(limit)

        with MBDB.get_adbc_connection() as conn:
            stmt = compile_sql(id_query)
            logger.debug(f"Query SQL: \n {stmt}")

            mb_ids = sql_to_df(
                stmt=stmt,
                connection=conn,
                return_type="arrow",
            )

        return mb_ids


def _build_unnested_clusters() -> CTE:
    """Create CTE that unnests cluster IDs for easier joining."""
    return (
        select(
            Clusters.cluster_id,
            ClusterSourceKey.source_config_id.label("source_config"),
            ClusterSourceKey.key,
        )
        .select_from(Clusters)
        .join(ClusterSourceKey, ClusterSourceKey.cluster_id == Clusters.cluster_id)
        .cte("unnested_clusters")
        .prefix_with("MATERIALIZED")
    )


def _find_source_cluster(
    unnested_clusters: CTE, source_config_id: int, key: str
) -> Select:
    """Find the initial cluster containing the source key."""
    return (
        select(unnested_clusters.c.cluster_id)
        .select_from(unnested_clusters)
        .where(
            and_(
                unnested_clusters.c.source_config == source_config_id,
                unnested_clusters.c.key == key,
            )
        )
        .scalar_subquery()
    )


def _build_hierarchy_up(
    source_cluster: Select,
    contains_table: CTE | Contains,
) -> CTE:
    """Build recursive CTE that finds all parent clusters.

    Args:
        source_cluster: Subquery that finds starting cluster
        contains_table: Contains or CTE of valid clusters to filter by
    """
    if isinstance(contains_table, CTE):
        child_col = contains_table.c.child
        parent_col = contains_table.c.parent
    else:
        child_col = contains_table.child
        child_col = contains_table.parent

    # Base case: direct parents
    base = (
        select(
            source_cluster.label("original_cluster"),
            source_cluster.label("child"),
            parent_col.label("parent"),
            literal(1).label("level"),
        )
        .select_from(contains_table)
        .where(child_col == source_cluster)
    )

    hierarchy_up = base.cte("hierarchy_up", recursive=True)

    # Recursive case
    recursive = (
        select(
            hierarchy_up.c.original_cluster,
            hierarchy_up.c.parent.label("child"),
            parent_col.label("parent"),
            (hierarchy_up.c.level + 1).label("level"),
        )
        .select_from(hierarchy_up)
        .join(contains_table, child_col == hierarchy_up.c.parent)
    )

    return hierarchy_up.union_all(recursive)


def _find_highest_parent(hierarchy_up: CTE) -> Select:
    """Find the topmost parent cluster from the hierarchy."""
    return (
        select(hierarchy_up.c.parent)
        .order_by(hierarchy_up.c.level.desc())
        .limit(1)
        .scalar_subquery()
    )


def _build_hierarchy_down(
    highest_parent: Select, unnested_clusters: CTE, contains_table: CTE | Contains
) -> CTE:
    """Build recursive CTE that finds all child clusters and their IDs.

    Args:
        highest_parent: Subquery that finds top cluster
        unnested_clusters: CTE with unnested cluster IDs
        contains_table: Contains or CTE of valid clusters to filter by
    """
    if isinstance(contains_table, CTE):
        child_col = contains_table.c.child
        parent_col = contains_table.c.parent
    else:
        child_col = contains_table.child
        child_col = contains_table.parent

    # Base case: Get both direct children and their IDs
    base = (
        select(
            highest_parent.label("parent"),
            child_col.label("child"),
            literal(1).label("level"),
            unnested_clusters.c.source_config.label("source_config"),
            unnested_clusters.c.key.label("key"),
        )
        .select_from(contains_table)
        .join_from(
            contains_table,
            unnested_clusters,
            unnested_clusters.c.cluster_id == child_col,
            isouter=True,
        )
        .where(parent_col == highest_parent)
    )

    hierarchy_down = base.cte("hierarchy_down", recursive=True)

    # Recursive case: Get both intermediate nodes AND their leaf records
    recursive = (
        select(
            hierarchy_down.c.parent,
            child_col.label("child"),
            (hierarchy_down.c.level + 1).label("level"),
            unnested_clusters.c.source_config.label("source_config"),
            unnested_clusters.c.key.label("key"),
        )
        .select_from(hierarchy_down)
        .join_from(
            hierarchy_down,
            contains_table,
            parent_col == hierarchy_down.c.child,
        )
        .join_from(
            contains_table,
            unnested_clusters,
            unnested_clusters.c.cluster_id == child_col,
            isouter=True,
        )
        .where(hierarchy_down.c.key.is_(None))  # Only recurse on non-leaf nodes
    )

    return hierarchy_down.union_all(recursive)


def _build_match_query(
    key: str,
    source_config: SourceConfigs,
    resolution: ResolutionName,
    session: Session,
    threshold: int | None = None,
) -> Select:
    """Builds the SQL query that powers the match function."""
    # Get truth resolution
    truth_resolution = (
        session.query(Resolutions).filter(Resolutions.name == resolution).first()
    )
    if truth_resolution is None:
        raise MatchboxResolutionNotFoundError(name=resolution)

    # Get resolution lineage and resolve thresholds
    lineage_truths = truth_resolution.get_lineage()
    thresholds = _resolve_thresholds(
        lineage_truths=lineage_truths,
        resolution=truth_resolution,
        threshold=threshold,
    )

    # Get valid clusters across all resolutions
    valid_clusters = _union_valid_clusters(thresholds)

    # Pre-filter Contains table if valid_clusters is provided
    contains_table = Contains
    if valid_clusters is not None:
        contains_table = _build_valid_contains(valid_clusters, name="valid_contains_up")

    # Build the query components
    unnested = _build_unnested_clusters()
    source_cluster = _find_source_cluster(unnested, source_config.source_config_id, key)
    hierarchy_up = _build_hierarchy_up(source_cluster, contains_table)
    highest = _find_highest_parent(hierarchy_up)
    hierarchy_down = _build_hierarchy_down(highest, unnested, contains_table)

    # Get all matched IDs
    final_stmt = (
        select(
            hierarchy_down.c.parent.label("cluster"),
            hierarchy_down.c.source_config,
            hierarchy_down.c.key,
        )
        .distinct()
        .select_from(hierarchy_down)
    )

    return final_stmt


def match(
    key: str,
    source: SourceResolutionName,
    targets: list[SourceResolutionName],
    resolution: ResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Matches an ID in a source resolution and returns the keys in the targets.

    To accomplish this, the function:

    * Reconstructs the resolution lineage from the specified resolution
    * Iterates through each target, and
        * Retrieves its cluster hash according to the resolution
        * Retrieves all other IDs in the cluster in the source source resolution
        * Retrieves all other IDs in the cluster in the target source resolution
    * Returns the results as Match objects, one per target
    """
    with MBDB.get_session() as session:
        # Get all matches for keys in all possible targets
        source_config = get_source_config(source, session)

        match_stmt = _build_match_query(
            key=key,
            source_config=source_config,
            resolution=resolution,
            session=session,
            threshold=threshold,
        )

        logger.debug(f"Match SQL: \n {compile_sql(match_stmt)}")

        matches = session.execute(match_stmt).all()

        # Return matches in target sources only
        cluster = None
        matches_by_source_id: dict[int, set] = {}
        for cluster_id, source_id, id_in_source in matches:
            if cluster is None:
                cluster = cluster_id
            if source_id not in matches_by_source_id:
                matches_by_source_id[source_id] = set()
            matches_by_source_id[source_id].add(id_in_source)

        result = []
        for target in targets:
            target_source = get_source_config(target, session)
            match_obj = Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_source_id.get(
                    source_config.source_config_id, set()
                ),
                target=target,
                target_id=matches_by_source_id.get(
                    target_source.source_config_id, set()
                ),
            )
            result.append(match_obj)

        return result
