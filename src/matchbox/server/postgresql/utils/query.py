"""Utilities for querying and matching in the PostgreSQL backend."""

from collections import defaultdict
from typing import Literal

import pyarrow as pa
from sqlalchemy import and_, func, join, literal_column, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import CTE, Select, Subquery

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Match,
    ResolverStepPath,
    SourceStepPath,
    StepType,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxStepNotQueriable,
    MatchboxStepTypeError,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ClusterSourceKey,
    Contains,
    ResolverClusters,
    SourceConfigs,
    Steps,
)
from matchbox.server.postgresql.utils.db import compile_sql


def _build_unified_query(
    step: Steps,
    sources: list[SourceConfigs] | None = None,
    level: Literal["leaf", "key"] = "leaf",
    include_source_config_id: bool = False,
) -> Select:
    """Build a query that projects records to root IDs through the hierarchy."""
    lineage = step.get_lineage(sources=sources, queryable_only=True)

    # Separate lineage entries into resolver steps and source config IDs.
    # Entries without a source_config_id are resolver nodes, the rest are sources
    resolver_ids: list[int] = []
    source_config_ids: list[int] = []
    for step_id, source_config_id in lineage:
        if source_config_id is None:
            resolver_ids.append(step_id)
        else:
            source_config_ids.append(source_config_id)

    # ClusterSourceKey is the base table, resolver subqueries are LEFT JOINed onto it
    from_clause = ClusterSourceKey
    projected_roots: list[ColumnElement[int]] = []

    for resolver_id in resolver_ids:
        # For each resolver, build a subquery that maps leaf cluster IDs to their
        # root cluster IDs at that step level via the Contains table.
        assignments: Subquery = (
            select(
                Contains.leaf.label("leaf_id"),
                Contains.root.label("root_id"),
            )
            .select_from(Contains)
            .join(
                ResolverClusters,
                and_(
                    ResolverClusters.cluster_id == Contains.root,
                    ResolverClusters.step_id == resolver_id,
                ),
            )
            .subquery(f"resolver_assignments_{resolver_id}")
        )
        # LEFT JOIN so that records not claimed by this resolver are still returned
        from_clause = join(
            from_clause,
            assignments,
            assignments.c.leaf_id == ClusterSourceKey.cluster_id,
            isouter=True,
        )
        projected_roots.append(assignments.c.root_id)

    # COALESCE across all resolver root columns
    # First non-null wins, giving higher-priority resolvers precedence
    # Falls back to the source cluster ID when no resolver has claimed the record
    root_projection: ColumnElement[int] = (
        func.coalesce(*projected_roots, ClusterSourceKey.cluster_id)
        if projected_roots
        else ClusterSourceKey.cluster_id
    )

    selection: list[ColumnElement] = [
        root_projection.label("root_id"),
        ClusterSourceKey.cluster_id.label("leaf_id"),
    ]
    if level == "key":
        # "key" level adds the source key, producing more rows than "leaf" because
        # multiple keys can share the same leaf cluster
        selection.append(ClusterSourceKey.key)
    if include_source_config_id:
        selection.append(ClusterSourceKey.source_config_id.label("source_config_id"))

    query_stmt = (
        select(*selection)
        .select_from(from_clause)
        .where(ClusterSourceKey.source_config_id.in_(source_config_ids))
    )

    # At "leaf" level, deduplicate rows introduced because multiple keys share
    # the same leaf cluster
    if level == "leaf":
        query_stmt = query_stmt.distinct()

    return query_stmt


def _build_target_cluster_cte(
    key: str,
    source_config_id: int,
    step: Steps,
) -> CTE:
    """Build the target cluster CTE for a source key."""
    # Reuse the unified query at key level, filtered to this source config,
    # so we get the resolved root cluster for the given key
    source_projection = _build_unified_query(
        step=step,
        level="key",
        include_source_config_id=True,
    ).subquery("source_projection")

    return (
        select(source_projection.c.root_id.label("cluster_id"))
        .where(
            and_(
                source_projection.c.source_config_id == source_config_id,
                source_projection.c.key == key,
            )
        )
        # Exactly one cluster per key
        # LIMIT 1 avoids a redundant scan
        .limit(1)
        .cte("target_cluster")
    )


def _build_matching_leaves_cte(
    source_and_target_ids: list[int],
    step: Steps,
    target_cluster_cte: CTE,
) -> CTE:
    """Build the matching keys CTE for a resolved cluster."""
    # Project all source + target keys through the hierarchy, then filter to those
    # whose resolved root matches the target cluster
    full_projection = _build_unified_query(
        step=step,
        level="key",
        include_source_config_id=True,
    ).subquery("full_projection")

    return (
        select(
            full_projection.c.root_id.label("cluster_id"),
            full_projection.c.source_config_id,
            full_projection.c.key,
        )
        .where(
            and_(
                full_projection.c.source_config_id.in_(source_and_target_ids),
                full_projection.c.root_id == target_cluster_cte.c.cluster_id,
            )
        )
        .distinct()
        .cte("matching_leaves")
    )


def require_complete_resolver(
    session: Session,
    path: ResolverStepPath,
) -> Steps:
    """Resolve and validate a resolver path for query-time operations."""
    resolver_step = Steps.from_path(path=path, session=session)
    if resolver_step.type != StepType.RESOLVER:
        raise MatchboxStepTypeError(
            step_name=str(path),
            step_type=resolver_step.type,
            expected_step_types=[StepType.RESOLVER],
        )
    if resolver_step.upload_stage != UploadStage.COMPLETE:
        raise MatchboxStepNotQueriable
    return resolver_step


def resolver_membership_subquery(
    step_id: int,
    alias: str = "resolver_membership",
) -> Subquery:
    """Build root_id/leaf_id membership rows for a resolver."""
    # First branch: root clusters count as their own leaf (self-membership)
    roots_query = select(
        ResolverClusters.cluster_id.label("root_id"),
        ResolverClusters.cluster_id.label("leaf_id"),
    ).where(ResolverClusters.step_id == step_id)

    # Second branch: all clusters contained within a root via the Contains table
    leaves_query = (
        select(
            ResolverClusters.cluster_id.label("root_id"),
            Contains.leaf.label("leaf_id"),
        )
        .select_from(ResolverClusters)
        .join(Contains, Contains.root == ResolverClusters.cluster_id)
        .where(ResolverClusters.step_id == step_id)
    )

    # UNION deduplicates in case a root cluster also appears as a leaf
    return roots_query.union(leaves_query).subquery(alias)


def query(
    source: SourceStepPath,
    resolves_from: ResolverStepPath | None = None,
    return_leaf_id: bool = False,
    limit: int | None = None,
) -> pa.Table:
    """Query Matchbox to retrieve linked data for a source."""
    with MBDB.get_session() as session:
        source_step: Steps = Steps.from_path(
            path=source,
            session=session,
        )
        source_config: SourceConfigs = source_step.source_config

        # Use the provided point-of-truth resolver, or fall back to the source step
        # for a simple self-contained query.
        if resolves_from is None:
            truth_step = source_step
        else:
            truth_step = require_complete_resolver(session, resolves_from)

        if truth_step.upload_stage != UploadStage.COMPLETE:
            raise MatchboxStepNotQueriable

        query_stmt = _build_unified_query(
            step=truth_step,
            sources=[source_config],
            level="key",
            include_source_config_id=False,
        )

    # Order outside the session so the sort is applied to the final statement
    query_stmt = query_stmt.order_by(
        literal_column("root_id"),
        literal_column("leaf_id"),
        ClusterSourceKey.key,
    )

    if limit is not None:
        query_stmt = query_stmt.limit(limit)

    with MBDB.get_adbc_connection() as conn:
        stmt = compile_sql(query_stmt)
        logger.debug(f"Query SQL: \n {stmt}")
        # Rename root_id → id to match the public-facing schema
        id_results = sql_to_df(
            stmt=stmt,
            connection=conn,
            return_type="arrow",
        ).rename_columns({"root_id": "id"})

    selection = ["id", "key"]
    if return_leaf_id:
        selection.append("leaf_id")

    return id_results.select(selection)


def match(
    key: str,
    source: SourceStepPath,
    targets: list[SourceStepPath],
    resolves_from: ResolverStepPath,
) -> list[Match]:
    """Match a source key against targets under a resolver point-of-truth."""
    with MBDB.get_session() as session:
        source_config: SourceConfigs = Steps.from_path(
            path=source,
            session=session,
        ).source_config
        resolver_step = require_complete_resolver(session, resolves_from)

        # Resolve source configs for all targets to enable ID lookup and result assembly
        target_configs: list[SourceConfigs] = [
            Steps.from_path(path=target, session=session).source_config
            for target in targets
        ]
        source_and_target_ids: list[int] = [
            source_config.source_config_id,
            *(tc.source_config_id for tc in target_configs),
        ]

        # Resolve which cluster this key belongs to under the point-of-truth
        target_cluster_cte = _build_target_cluster_cte(
            key=key,
            source_config_id=source_config.source_config_id,
            step=resolver_step,
        )

        matching_leaves_cte = _build_matching_leaves_cte(
            source_and_target_ids=source_and_target_ids,
            step=resolver_step,
            target_cluster_cte=target_cluster_cte,
        )

        matched_rows = session.execute(
            select(
                matching_leaves_cte.c.cluster_id,
                matching_leaves_cte.c.source_config_id,
                matching_leaves_cte.c.key,
            )
        ).all()

        # Accumulate matching keys by source config ID for fast lookup below
        cluster: int | None = None
        matches_by_source_id: defaultdict[int, set[str]] = defaultdict(set)
        for cluster_id, source_config_id_result, key_in_source in matched_rows:
            if cluster is None:
                cluster = cluster_id
            matches_by_source_id[source_config_id_result].add(key_in_source)

        # Build one Match object per target, defaulting to an empty set when no
        # keys were found for that target config
        return [
            Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_source_id[source_config.source_config_id],
                target=target,
                target_id=matches_by_source_id[target_config.source_config_id],
            )
            for target, target_config in zip(targets, target_configs, strict=False)
        ]
